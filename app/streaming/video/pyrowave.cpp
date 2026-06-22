// PyroWave wavelet video decoder for moonlight-qt.
//
// Decodes a contiguous PyroWave bitstream (one frame per submit) with Granite +
// PyroWave on Vulkan and presents straight to the SDL window's Vulkan swapchain
// (no CPU readback). The chroma subsampling (4:2:0 / 4:4:4) and color volume
// (SDR BT.709, SDR BT.2020, HDR PQ BT.2020) are detected per stream from the
// bitstream sequence header, so a single code path covers every negotiated
// PyroWave format. A YCbCr->RGB present pass (yuv2rgb.frag) does the conversion
// on the GPU, driven by specialization constants, into an SDR (UNORM) or HDR10
// (ST2084) swapchain.

#include "pyrowave.h"

// Qt defines `signals` as a macro which conflicts with Granite's member of the
// same name. Undefine it around the Granite/PyroWave headers.
#pragma push_macro("signals")
#undef signals

// Granite Vulkan
#include "context.hpp"
#include "device.hpp"
#include "wsi.hpp"
#include "command_buffer.hpp"
#include "image.hpp"
#include "thread_id.hpp"

// PyroWave
#include "pyrowave_decoder.hpp"
#include "pyrowave_config.hpp"
#include "pyrowave_common.hpp"

#pragma pop_macro("signals")

// SDL / Qt
#include <SDL.h>
#include <SDL_vulkan.h>

// moonlight-common
#include <Limelight.h>

#include <cstring>
#include <vector>

using namespace Vulkan;

// Granite keys its per-thread command pools off a thread-local index that is
// only set on threads it spawned (or the main thread). moonlight calls us from
// its own decoder thread, which Granite doesn't know about, so every Device
// operation logs "Thread does not exist in thread manager...". We are the only
// thread doing Granite work for this Device, so claim index 0. Call this on
// entry to every Granite-touching method (cheap thread-local write, guarded).
static void claimGraniteThread()
{
    static thread_local bool registered = false;
    if (!registered) {
        Util::register_thread_index(0);
        registered = true;
    }
}

// SPIR-V for the present pass (compiled offline, embedded as C arrays). Shared
// verbatim with the Android JNI decoder.
static const uint32_t fullscreen_vert_spv[] =
#include "pyrowave_shaders/fullscreen.vert.inc"
;
static const uint32_t yuv2rgb_frag_spv[] =
#include "pyrowave_shaders/yuv2rgb.frag.inc"
;

// ---------------------------------------------------------------------------
// WSI platform backed by an SDL window (the same surface the rest of moonlight
// would use for its Vulkan renderer). The window is created with
// SDL_WINDOW_VULKAN on platforms where libplacebo/Vulkan is available.
// ---------------------------------------------------------------------------
namespace {
class SdlWSIPlatform : public WSIPlatform {
public:
    void set_window(SDL_Window *w) { window = w; }

    VkSurfaceKHR create_surface(VkInstance instance, VkPhysicalDevice) override {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (!SDL_Vulkan_CreateSurface(window, instance, &surface)) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "PyroWave: SDL_Vulkan_CreateSurface() failed: %s", SDL_GetError());
            return VK_NULL_HANDLE;
        }
        return surface;
    }

    std::vector<const char *> get_instance_extensions() override {
        return queryInstanceExtensions(window);
    }

    uint32_t get_surface_width() override  { int w = 0, h = 0; if (window) SDL_Vulkan_GetDrawableSize(window, &w, &h); return (uint32_t) w; }
    uint32_t get_surface_height() override { int w = 0, h = 0; if (window) SDL_Vulkan_GetDrawableSize(window, &w, &h); return (uint32_t) h; }

    bool alive(WSI &) override { return window != nullptr; }
    void poll_input() override {}
    void poll_input_async(Granite::InputTrackerHandler *) override {}

    static std::vector<const char *> queryInstanceExtensions(SDL_Window *window) {
        unsigned count = 0;
        if (!window || !SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr))
            return {};
        std::vector<const char *> exts(count);
        if (count && !SDL_Vulkan_GetInstanceExtensions(window, &count, exts.data()))
            return {};
        return exts;
    }

private:
    SDL_Window *window = nullptr;
};
} // namespace

// ---------------------------------------------------------------------------

struct PyroWaveVideoDecoder::Impl {
    SDL_Window *window = nullptr;
    SdlWSIPlatform platform;
    WSI wsi;
    Device *device = nullptr;

    int width = 0, height = 0, chromaW = 0, chromaH = 0;
    PyroWave::ChromaSubsampling chroma = PyroWave::ChromaSubsampling::Chroma420;

    // Color metadata decoded from the BitstreamSequenceHeader on the first frame.
    bool is_hdr = false;      // PQ transfer  -> drive an HDR10 swapchain
    bool bt2020 = false;      // BT.2020 primaries/matrix
    bool full_range = false;  // JPEG (full) vs MPEG (limited) range

    bool decoder_ready = false;
    bool swapchain_ready = false;
    bool testOnly = false;

    PyroWave::Decoder decoder;
    ImageHandle yuvImages[3];
    PyroWave::ViewBuffers views;
    Program *present_program = nullptr;

    // Reassembly scratch for a frame's LENTRY chain (see submitDecodeUnit).
    std::vector<uint8_t> packetScratch;

    bool init_swapchain(bool want_hdr);
    bool init_decoder(PyroWave::ChromaSubsampling c);
};

// ---------------------------------------------------------------------------
// Construction / destruction

PyroWaveVideoDecoder::PyroWaveVideoDecoder()
    : d(std::make_unique<Impl>()) {}

PyroWaveVideoDecoder::~PyroWaveVideoDecoder()
{
    claimGraniteThread();
    if (d->device)
        d->device->wait_idle();
    for (auto &img : d->yuvImages)
        img.reset();
    // d->wsi tears down the swapchain and device on destruction.
}

// ---------------------------------------------------------------------------

bool PyroWaveVideoDecoder::Impl::init_swapchain(bool want_hdr)
{
    // HDR10 needs the ST2084 color space; fall back to UNORM SDR otherwise. If
    // the surface can't provide HDR10, retry as SDR so the present shader still
    // outputs sensible (clipped) SDR rather than failing the stream.
    wsi.set_backbuffer_format(want_hdr ? BackbufferFormat::HDR10 : BackbufferFormat::UNORM);
    if (!wsi.init_surface_swapchain()) {
        if (want_hdr) {
            SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION,
                        "PyroWave: HDR10 swapchain init failed, retrying as SDR");
            wsi.set_backbuffer_format(BackbufferFormat::UNORM);
            if (!wsi.init_surface_swapchain()) {
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: swapchain init failed");
                return false;
            }
            is_hdr = false;
        } else {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: swapchain init failed");
            return false;
        }
    }
    swapchain_ready = true;
    return true;
}

bool PyroWaveVideoDecoder::Impl::init_decoder(PyroWave::ChromaSubsampling c)
{
    chroma  = c;
    chromaW = (c == PyroWave::ChromaSubsampling::Chroma420) ? width >> 1 : width;
    chromaH = (c == PyroWave::ChromaSubsampling::Chroma420) ? height >> 1 : height;

    // 10-bit HDR decodes into R16_UNORM planes; SDR stays on R8_UNORM. The
    // present pass needs to sample the planes (SAMPLED) and the fragment decode
    // path renders into them (COLOR_ATTACHMENT).
    VkFormat plane_format = is_hdr ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
    ImageCreateInfo info = ImageCreateInfo::immutable_2d_image(width, height, plane_format);
    info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_STORAGE_BIT;
    info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    yuvImages[0] = device->create_image(info);
    info.width = chromaW; info.height = chromaH;
    yuvImages[1] = device->create_image(info);
    yuvImages[2] = device->create_image(info);
    if (!yuvImages[0] || !yuvImages[1] || !yuvImages[2]) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: failed to allocate YCbCr plane images");
        return false;
    }
    for (int i = 0; i < 3; i++)
        views.planes[i] = &yuvImages[i]->get_view();

    // Use the fragment iDWT path: decode() then leaves the planes in
    // SHADER_READ_ONLY_OPTIMAL, ready for the present pass to sample without an
    // extra barrier. This matches the Android decoder and works on any GPU.
    if (!decoder.init(device, width, height, chroma, /*fragment_path*/ true)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: Decoder::init() failed");
        return false;
    }

    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                "PyroWave decoder ready %dx%d chroma=%s hdr=%d bt2020=%d full_range=%d",
                width, height,
                (c == PyroWave::ChromaSubsampling::Chroma444) ? "4:4:4" : "4:2:0",
                is_hdr, bt2020, full_range);
    decoder_ready = true;
    return true;
}

bool PyroWaveVideoDecoder::initialize(PDECODER_PARAMETERS params)
{
    d->window   = params->window;
    d->width    = params->width;
    d->height   = params->height;
    d->testOnly = params->testOnly;

    claimGraniteThread();

    if (!Context::init_loader(nullptr)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: Vulkan loader init failed");
        return false;
    }

    // Decoder probing only needs to confirm a Vulkan device can be created. The
    // probe's test window is hidden and may not be a Vulkan window at all (a
    // hidden SDL_WINDOW_VULKAN can fail to create and fall back to a plain
    // window), so go headless here — no surface/swapchain. The real stream window
    // (created with SDL_WINDOW_VULKAN) provides the surface for actual playback.
    if (d->testOnly) {
        ContextHandle probeCtx = Util::make_handle<Context>();
        if (!probeCtx->init_instance_and_device(nullptr, 0, nullptr, 0,
                                                CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT)) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "PyroWave: headless Vulkan context init failed (probe)");
            return false;
        }
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: probe OK (headless Vulkan device)");
        return true;
    }

    d->platform.set_window(d->window);

    // Make sure SDL's Vulkan library is loaded before querying instance
    // extensions. A window created with SDL_WINDOW_VULKAN normally auto-loads it,
    // but sdl2-compat (SDL2 API on an SDL3 runtime) does not always, and
    // SDL_Vulkan_GetInstanceExtensions() then reports no extensions. Loading it
    // explicitly is a no-op refcount bump if it is already loaded.
    if (SDL_Vulkan_LoadLibrary(nullptr) != 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION,
                    "PyroWave: SDL_Vulkan_LoadLibrary() failed: %s", SDL_GetError());
    }

    // Build the instance with the surface extensions SDL needs for this window
    // (VK_KHR_surface + the platform-specific surface extension), so the surface
    // we create later via SDL_Vulkan_CreateSurface() is valid for it.
    std::vector<const char *> iext = SdlWSIPlatform::queryInstanceExtensions(d->window);
    if (iext.empty()) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "PyroWave: SDL_Vulkan_GetInstanceExtensions() returned none "
                     "(window flags=0x%x, SDL error: %s)",
                     (unsigned) SDL_GetWindowFlags(d->window), SDL_GetError());
        return false;
    }
    const char *dext[] = { "VK_KHR_swapchain" };

    ContextHandle ctx = Util::make_handle<Context>();
    if (!ctx->init_instance_and_device(iext.data(), (uint32_t) iext.size(), dext, 1,
                                       CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: Granite Vulkan context init failed");
        return false;
    }

    d->wsi.set_platform(&d->platform);
    // Mailbox (newest-frame-wins, no tearing): decode never blocks on vsync, the
    // display always shows the latest decoded frame, no upstream queue buildup.
    d->wsi.set_present_mode(PresentMode::UnlockedNoTearing);
    // Default to SDR; init_swapchain() switches to HDR10 once the first frame's
    // sequence header reveals a PQ transfer function.
    d->wsi.set_backbuffer_format(BackbufferFormat::UNORM);
    if (!d->wsi.init_from_existing_context(std::move(ctx))) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: WSI context init failed");
        return false;
    }
    if (!d->wsi.init_device()) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: WSI device init failed");
        return false;
    }
    d->device = &d->wsi.get_device();

    d->present_program = d->device->request_program(
        fullscreen_vert_spv, sizeof(fullscreen_vert_spv),
        yuv2rgb_frag_spv,    sizeof(yuv2rgb_frag_spv));
    if (!d->present_program) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: present program creation failed");
        return false;
    }

    // For decoder probing we only need to prove the Vulkan context comes up; the
    // swapchain and decoder are created lazily on the first real frame once the
    // stream format is known.
    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                "PyroWave Vulkan context ready %dx%d (format detected from first frame)",
                d->width, d->height);
    return true;
}

QSize PyroWaveVideoDecoder::getDecoderMaxResolution()
{
    return QSize(4096, 4096);
}

int PyroWaveVideoDecoder::getDecoderColorspace()
{
    // COLORSPACE_REC_709 (1) / COLORSPACE_REC_2020 (2)
    return d->bt2020 ? COLORSPACE_REC_2020 : COLORSPACE_REC_709;
}

int PyroWaveVideoDecoder::getDecoderColorRange()
{
    // COLOR_RANGE_LIMITED (0) / COLOR_RANGE_FULL (1)
    return d->full_range ? COLOR_RANGE_FULL : COLOR_RANGE_LIMITED;
}

void PyroWaveVideoDecoder::setHdrMode(bool)
{
    // No-op: HDR vs SDR is detected per stream from the bitstream sequence
    // header (see submitDecodeUnit) and drives the swapchain color space there.
}

// ---------------------------------------------------------------------------

int PyroWaveVideoDecoder::submitDecodeUnit(PDECODE_UNIT du)
{
    claimGraniteThread();

    // The PyroWave bitstream is a contiguous stream of self-describing packets,
    // emitted by the server as one blob per frame. moonlight-common-c splits that
    // blob into an LENTRY chain whose chunk boundaries do NOT respect PyroWave
    // packet boundaries, so feeding each LENTRY separately makes push_packet read
    // a header that runs past the end of the chunk. Reassemble the whole frame
    // into one contiguous buffer and push it once.
    const uint8_t *frameData;
    size_t frameLen;
    if (du->bufferList->next == nullptr) {
        frameData = reinterpret_cast<const uint8_t *>(du->bufferList->data);
        frameLen = (size_t) du->bufferList->length;
    } else {
        d->packetScratch.clear();
        d->packetScratch.reserve((size_t) du->fullLength);
        for (PLENTRY entry = du->bufferList; entry != nullptr; entry = entry->next) {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(entry->data);
            d->packetScratch.insert(d->packetScratch.end(), p, p + entry->length);
        }
        frameData = d->packetScratch.data();
        frameLen = d->packetScratch.size();
    }

    // Decoder probing only validates initialization; never decode/present.
    if (d->testOnly)
        return DR_OK;

    // First frame: detect chroma subsampling and color volume from the bitstream
    // sequence header (always the first bytes of a frame, extended=1), then bring
    // up the matching swapchain and decoder.
    if (!d->decoder_ready) {
        if (frameLen < sizeof(PyroWave::BitstreamSequenceHeader))
            return DR_NEED_IDR;
        auto *seq = reinterpret_cast<const PyroWave::BitstreamSequenceHeader *>(frameData);
        if (!seq->extended)
            return DR_NEED_IDR;

        auto detected = (seq->chroma_resolution == PyroWave::CHROMA_RESOLUTION_444)
                            ? PyroWave::ChromaSubsampling::Chroma444
                            : PyroWave::ChromaSubsampling::Chroma420;
        d->bt2020     = seq->color_primaries == PyroWave::COLOR_PRIMARIES_BT2020;
        d->full_range = seq->ycbcr_range == PyroWave::YCBCR_RANGE_FULL;
        d->is_hdr     = seq->transfer_function == PyroWave::TRANSFER_FUNCTION_PQ;

        if (!d->init_swapchain(d->is_hdr))
            return DR_NEED_IDR;
        if (!d->init_decoder(detected))
            return DR_NEED_IDR;
    }

    if (!d->decoder.push_packet(frameData, frameLen)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: push_packet() failed");
        d->decoder.clear();
        return DR_NEED_IDR;
    }

    // Accumulating; not a full frame yet.
    if (!d->decoder.decode_is_ready(false))
        return DR_OK;

    if (!d->wsi.begin_frame())
        return DR_OK;

    auto cmd = d->device->request_command_buffer();

    // decode() handles all output-plane layout transitions internally
    // (UNDEFINED -> COLOR_ATTACHMENT -> SHADER_READ_ONLY for the fragment path).
    if (!d->decoder.decode(*cmd, d->views)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "PyroWave: decode() failed");
        d->device->submit(cmd);
        d->wsi.end_frame();
        return DR_NEED_IDR;
    }

    // Present pass: sample the 3 planes, YCbCr->RGB into the swapchain image.
    cmd->begin_render_pass(d->device->get_swapchain_render_pass(SwapchainRenderPass::ColorOnly));
    cmd->set_quad_state();
    cmd->set_program(d->present_program);
    cmd->set_texture(0, 0, *d->views.planes[0]);
    cmd->set_texture(0, 1, *d->views.planes[1]);
    cmd->set_texture(0, 2, *d->views.planes[2]);
    cmd->set_sampler(0, 3, StockSampler::LinearClamp);
    // Spec constants drive the YCbCr->RGB conversion variant in yuv2rgb.frag:
    //   0: FullRange (JPEG vs MPEG range)
    //   1: BT2020    (BT.2020 vs BT.709 matrix)
    //   2: HdrPQ     (output PQ BT.2020 for an HDR10 swapchain)
    cmd->set_specialization_constant_mask(0x7);
    cmd->set_specialization_constant(0, d->full_range);
    cmd->set_specialization_constant(1, d->bt2020);
    cmd->set_specialization_constant(2, d->is_hdr);
    cmd->draw(3);
    cmd->end_render_pass();

    d->device->submit(cmd);
    d->wsi.end_frame();
    return DR_OK;
}

// ---------------------------------------------------------------------------

void PyroWaveVideoDecoder::renderFrameOnMainThread()
{
    // No-op: submitDecodeUnit() decodes and presents to the Vulkan swapchain on
    // the decoder thread. Nothing to do on the main thread.
}
