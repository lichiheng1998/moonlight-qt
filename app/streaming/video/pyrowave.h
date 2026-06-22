#pragma once

#include "decoder.h"
#include <memory>

class PyroWaveVideoDecoder : public IVideoDecoder {
public:
    PyroWaveVideoDecoder();
    ~PyroWaveVideoDecoder();

    bool initialize(PDECODER_PARAMETERS params) override;
    bool isHardwareAccelerated() override { return true; }
    bool isAlwaysFullScreen() override { return false; }
    // PyroWave presents through a Granite Vulkan swapchain that can request an
    // HDR10 (ST2084) color space, so advertise HDR support. The actual HDR vs
    // SDR path is chosen per-stream from the bitstream sequence header, with an
    // automatic SDR fallback if the surface can't provide HDR10.
    bool isHdrSupported() override { return true; }
    int getDecoderCapabilities() override { return 0; }
    int getDecoderColorspace() override; // COLORSPACE_REC_* (depends on stream)
    int getDecoderColorRange() override;  // COLOR_RANGE_* (depends on stream)
    QSize getDecoderMaxResolution() override;
    int submitDecodeUnit(PDECODE_UNIT du) override;
    void renderFrameOnMainThread() override;
    void setHdrMode(bool enabled) override;
    // Handle window state changes (size/fullscreen/display) WITHOUT a full
    // decoder recreation. The Granite WSI re-queries the surface size each frame
    // (SDL_Vulkan_GetDrawableSize) and recreates the swapchain on a dimension
    // change, so a resize/fullscreen transition needs no work here. Returning
    // false would make the session destroy and rebuild the whole decoder (a new
    // Granite context + swapchain on an already-fullscreen window), risking a
    // present that never recovers.
    bool notifyWindowChanged(PWINDOW_STATE_CHANGE_INFO) override { return true; }

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
