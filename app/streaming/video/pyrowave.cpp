// PyroWave wavelet video decoder for moonlight-qt
// GPU decode via Granite Vulkan + CPU readback into SDL YUV texture

#include "pyrowave.h"

// Qt defines `signals` as a macro which conflicts with Granite's member
// of the same name. Undefine it around the Granite/PyroWave headers.
#pragma push_macro("signals")
#undef signals

// Granite Vulkan
#include "context.hpp"
#include "device.hpp"
#include "command_buffer.hpp"
#include "buffer.hpp"
#include "image.hpp"

// PyroWave
#include "pyrowave_decoder.hpp"
#include "pyrowave_config.hpp"

#pragma pop_macro("signals")

// SDL / Qt
#include <SDL.h>
#include <QSize>

// moonlight-common
#include <Limelight.h>

#include <cstring>
#include <vector>

// -------------------------------------------------------------------------

struct PyroWaveVideoDecoder::Impl {
    // Destruction order is reverse of declaration order.
    // ctx/dev must outlive everything; decoder must outlive nothing (it uses images/staging during cleanup).
    Vulkan::Context ctx;
    Vulkan::Device dev;

    // GPU output planes (Y, Cb, Cr as R8_UNORM images)
    Vulkan::ImageHandle yuvImages[3];
    PyroWave::ViewBuffers views;

    // Host-visible staging buffers for GPU→CPU readback
    Vulkan::BufferHandle staging[3];

    // Decoder declared after images/staging so it is destroyed first,
    // while the images and staging buffers it references are still valid.
    PyroWave::Decoder decoder;

    // CPU-side decoded planes (written under mutex, read by renderFrameOnMainThread)
    std::vector<uint8_t> pendingY, pendingCb, pendingCr;

    // Reusable scratch buffer to reassemble a frame's LENTRY chain into one
    // contiguous blob before feeding it to push_packet (see submitDecodeUnit).
    std::vector<uint8_t> packetScratch;

    int width = 0;
    int height = 0;
    int chromaW = 0;
    int chromaH = 0;
    PyroWave::ChromaSubsampling chroma = PyroWave::ChromaSubsampling::Chroma420;

    // After the first decode the images are left in TRANSFER_SRC_OPTIMAL;
    // subsequent frames must transition from that layout back to GENERAL.
    bool imagesInitialized = false;

    bool testOnly = false;
    bool frameReady = false;

    SDL_Renderer *renderer = nullptr;
    SDL_Texture  *texture  = nullptr;
    SDL_mutex    *mutex    = nullptr;
};

// -------------------------------------------------------------------------
// Construction / destruction

PyroWaveVideoDecoder::PyroWaveVideoDecoder()
    : d(std::make_unique<Impl>()) {}

PyroWaveVideoDecoder::~PyroWaveVideoDecoder()
{
    if (d->texture)  { SDL_DestroyTexture(d->texture);   d->texture  = nullptr; }
    if (d->renderer) { SDL_DestroyRenderer(d->renderer); d->renderer = nullptr; }
    if (d->mutex)    { SDL_DestroyMutex(d->mutex);       d->mutex    = nullptr; }
}

// -------------------------------------------------------------------------

bool PyroWaveVideoDecoder::initialize(PDECODER_PARAMETERS params)
{
    d->width   = params->width;
    d->height  = params->height;
    d->testOnly = params->testOnly;

    // PyroWave only supports 4:2:0 for now (gaming content)
    d->chroma  = PyroWave::ChromaSubsampling::Chroma420;
    d->chromaW = params->width  >> 1;
    d->chromaH = params->height >> 1;

    // ---- Granite Vulkan context ----------------------------------------
    if (!Vulkan::Context::init_loader(nullptr)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: Vulkan loader init failed");
        return false;
    }
    if (!d->ctx.init_instance_and_device(
            nullptr, 0, nullptr, 0,
            Vulkan::CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: Granite Vulkan context init failed");
        return false;
    }
    d->dev.set_context(d->ctx);

    // ---- Output YCbCr images -------------------------------------------
    Vulkan::ImageCreateInfo imgInfo =
        Vulkan::ImageCreateInfo::immutable_2d_image(
            d->width, d->height, VK_FORMAT_R8_UNORM);
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT  |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    d->yuvImages[0] = d->dev.create_image(imgInfo);
    d->dev.set_name(*d->yuvImages[0], "pyrowave_dec_Y");

    imgInfo.width  = d->chromaW;
    imgInfo.height = d->chromaH;
    d->yuvImages[1] = d->dev.create_image(imgInfo);
    d->dev.set_name(*d->yuvImages[1], "pyrowave_dec_Cb");
    d->yuvImages[2] = d->dev.create_image(imgInfo);
    d->dev.set_name(*d->yuvImages[2], "pyrowave_dec_Cr");

    for (int i = 0; i < 3; i++)
        d->views.planes[i] = &d->yuvImages[i]->get_view();

    // ---- Init PyroWave decoder -----------------------------------------
    if (!d->decoder.init(&d->dev, d->width, d->height, d->chroma)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: Decoder::init() failed");
        return false;
    }

    // ---- Staging buffers for host readback -----------------------------
    auto makeStagingBuf = [&](VkDeviceSize sz) -> Vulkan::BufferHandle {
        Vulkan::BufferCreateInfo bi = {};
        bi.domain = Vulkan::BufferDomain::CachedHost;
        bi.usage  = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.size   = sz;
        return d->dev.create_buffer(bi);
    };
    d->staging[0] = makeStagingBuf((VkDeviceSize)d->width  * d->height);
    d->staging[1] = makeStagingBuf((VkDeviceSize)d->chromaW * d->chromaH);
    d->staging[2] = makeStagingBuf((VkDeviceSize)d->chromaW * d->chromaH);

    d->pendingY.resize((size_t)d->width   * d->height);
    d->pendingCb.resize((size_t)d->chromaW * d->chromaH);
    d->pendingCr.resize((size_t)d->chromaW * d->chromaH);

    d->mutex = SDL_CreateMutex();
    if (!d->mutex) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: SDL_CreateMutex() failed: %s", SDL_GetError());
        return false;
    }

    if (d->testOnly)
        return true;

    // ---- SDL renderer + YUV texture ------------------------------------
    // Mailbox / newest-frame-wins semantics (the Android JNI path achieves this
    // via Granite WSI PresentMode::UnlockedNoTearing == VK_PRESENT_MODE_MAILBOX).
    // That present-mode knob does not apply here because this path does not own a
    // Vulkan swapchain -- it decodes on the GPU, reads back into a SINGLE-SLOT CPU
    // buffer (pendingY/Cb/Cr, overwritten each decode so the newest frame always
    // wins, no queue buildup) and presents via SDL on the main thread. Crucially
    // the renderer is created WITHOUT SDL_RENDERER_PRESENTVSYNC, so SDL_RenderPresent
    // never blocks the decode thread on vsync. Decode and present run on separate
    // threads, so the Android FIFO problem (present blocking decode -> upstream
    // decode-unit queue fills -> dropped frames -> stutter) cannot occur here. The
    // mailbox optimization is therefore already in effect; no code change is needed.
    d->renderer = SDL_CreateRenderer(params->window, -1, SDL_RENDERER_ACCELERATED);
    if (!d->renderer) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: SDL_CreateRenderer() failed: %s", SDL_GetError());
        return false;
    }
    SDL_SetRenderDrawColor(d->renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(d->renderer);
    SDL_RenderPresent(d->renderer);

    // IYUV = planar I420 (Y plane, then Cb, then Cr)
    d->texture = SDL_CreateTexture(d->renderer,
                                    SDL_PIXELFORMAT_IYUV,
                                    SDL_TEXTUREACCESS_STREAMING,
                                    d->width, d->height);
    if (!d->texture) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: SDL_CreateTexture() failed: %s", SDL_GetError());
        return false;
    }
    SDL_SetTextureBlendMode(d->texture, SDL_BLENDMODE_NONE);

    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                "PyroWave decoder initialized (%dx%d)", d->width, d->height);
    return true;
}

QSize PyroWaveVideoDecoder::getDecoderMaxResolution()
{
    return QSize(4096, 4096);
}

// -------------------------------------------------------------------------

int PyroWaveVideoDecoder::submitDecodeUnit(PDECODE_UNIT du)
{
    // The PyroWave bitstream is a contiguous stream of self-describing packets,
    // emitted by the server as one blob per frame. moonlight-common-c splits that
    // blob into an LENTRY chain whose chunk boundaries do NOT respect PyroWave
    // packet boundaries, so feeding each LENTRY separately makes push_packet read
    // a header that runs past the end of the chunk ("Packet header states N bytes,
    // but only M bytes left to parse"). Reassemble the whole frame into one
    // contiguous buffer and push it once.
    const uint8_t *frameData;
    size_t frameLen;
    if (du->bufferList->next == nullptr) {
        // Single entry: feed directly, no copy needed.
        frameData = reinterpret_cast<const uint8_t *>(du->bufferList->data);
        frameLen = (size_t)du->bufferList->length;
    } else {
        d->packetScratch.clear();
        d->packetScratch.reserve((size_t)du->fullLength);
        for (PLENTRY entry = du->bufferList; entry != nullptr; entry = entry->next) {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(entry->data);
            d->packetScratch.insert(d->packetScratch.end(), p, p + entry->length);
        }
        frameData = d->packetScratch.data();
        frameLen = d->packetScratch.size();
    }

    if (!d->decoder.push_packet(frameData, frameLen)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: push_packet() failed");
        return DR_NEED_IDR;
    }

    // Wait until the decoder has enough data for a full frame
    if (!d->decoder.decode_is_ready(false))
        return DR_OK;

    // ---- GPU decode ----------------------------------------------------
    auto cmd = d->dev.request_command_buffer();

    // Transition images to VK_IMAGE_LAYOUT_GENERAL for the compute decode.
    // Old layout: UNDEFINED for the first frame, TRANSFER_SRC_OPTIMAL thereafter.
    for (int i = 0; i < 3; i++) {
        VkImageLayout oldLayout = d->imagesInitialized
                                  ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                  : VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 oldStage = d->imagesInitialized
                                         ? VK_PIPELINE_STAGE_2_COPY_BIT
                                         : VK_PIPELINE_STAGE_2_COPY_BIT;
        cmd->image_barrier(*d->yuvImages[i],
                           oldLayout, VK_IMAGE_LAYOUT_GENERAL,
                           oldStage, (VkAccessFlags2)0,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    d->decoder.decode(*cmd, d->views);

    // Transition GENERAL → TRANSFER_SRC_OPTIMAL for the buffer copy
    for (int i = 0; i < 3; i++) {
        cmd->image_barrier(*d->yuvImages[i],
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    d->imagesInitialized = true;

    // Copy each plane to its staging buffer
    int widths[3]  = { d->width,   d->chromaW, d->chromaW };
    int heights[3] = { d->height,  d->chromaH, d->chromaH };
    for (int i = 0; i < 3; i++) {
        VkExtent3D ext = { (uint32_t)widths[i], (uint32_t)heights[i], 1 };
        VkImageSubresourceLayers sub = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        cmd->copy_image_to_buffer(*d->staging[i], *d->yuvImages[i],
                                   0, {}, ext, 0, 0, sub);
    }

    // Barrier: ensure copy is visible to the host
    cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_HOST_BIT,   VK_ACCESS_HOST_READ_BIT);

    Vulkan::Fence fence;
    d->dev.submit(cmd, &fence);
    d->dev.next_frame_context();
    fence->wait();

    // ---- CPU readback --------------------------------------------------
    SDL_LockMutex(d->mutex);

    auto *y  = static_cast<const uint8_t *>(
        d->dev.map_host_buffer(*d->staging[0], Vulkan::MEMORY_ACCESS_READ_BIT));
    auto *cb = static_cast<const uint8_t *>(
        d->dev.map_host_buffer(*d->staging[1], Vulkan::MEMORY_ACCESS_READ_BIT));
    auto *cr = static_cast<const uint8_t *>(
        d->dev.map_host_buffer(*d->staging[2], Vulkan::MEMORY_ACCESS_READ_BIT));

    memcpy(d->pendingY.data(),  y,  d->pendingY.size());
    memcpy(d->pendingCb.data(), cb, d->pendingCb.size());
    memcpy(d->pendingCr.data(), cr, d->pendingCr.size());

    d->dev.unmap_host_buffer(*d->staging[0], Vulkan::MEMORY_ACCESS_READ_BIT);
    d->dev.unmap_host_buffer(*d->staging[1], Vulkan::MEMORY_ACCESS_READ_BIT);
    d->dev.unmap_host_buffer(*d->staging[2], Vulkan::MEMORY_ACCESS_READ_BIT);

    d->frameReady = true;
    SDL_UnlockMutex(d->mutex);

    // Signal the main thread to render
    if (!d->testOnly) {
        SDL_Event ev;
        ev.type      = SDL_USEREVENT;
        ev.user.code = SDL_CODE_FRAME_READY;
        SDL_PushEvent(&ev);
    }

    return DR_OK;
}

// -------------------------------------------------------------------------

void PyroWaveVideoDecoder::renderFrameOnMainThread()
{
    if (d->testOnly || !d->renderer || !d->texture)
        return;

    SDL_LockMutex(d->mutex);
    if (!d->frameReady) {
        SDL_UnlockMutex(d->mutex);
        return;
    }

    // BT.709 limited-range (default for gaming content)
    SDL_SetYUVConversionMode(SDL_YUV_CONVERSION_BT709);

    SDL_UpdateYUVTexture(d->texture, nullptr,
                         d->pendingY.data(),  d->width,
                         d->pendingCb.data(), d->chromaW,
                         d->pendingCr.data(), d->chromaW);
    d->frameReady = false;
    SDL_UnlockMutex(d->mutex);

    // Scale to fill window while preserving aspect ratio
    SDL_Rect src = { 0, 0, d->width, d->height };
    SDL_Rect dst = { 0, 0, 0, 0 };
    SDL_GetRendererOutputSize(d->renderer, &dst.w, &dst.h);

    SDL_RenderClear(d->renderer);
    SDL_RenderCopy(d->renderer, d->texture, &src, &dst);
    SDL_RenderPresent(d->renderer);
}
