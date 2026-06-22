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
    bool isHdrSupported() override { return false; }
    int getDecoderCapabilities() override { return 0; }
    int getDecoderColorspace() override { return 1; } // COLORSPACE_REC_709
    int getDecoderColorRange() override { return 0; } // COLOR_RANGE_LIMITED
    QSize getDecoderMaxResolution() override;
    int submitDecodeUnit(PDECODE_UNIT du) override;
    void renderFrameOnMainThread() override;
    void setHdrMode(bool) override {}
    // Handle window state changes (size/fullscreen/display) WITHOUT a full
    // decoder recreation. renderFrameOnMainThread() re-queries the renderer
    // output size every frame, so a resize/fullscreen transition needs no work
    // here. Returning false makes the session destroy and rebuild the whole
    // decoder (new Granite context + a new SDL_Renderer created while the window
    // is already fullscreen), which leaves the fullscreen renderer not presenting
    // (frozen on the first frame) even though decode/present keep running.
    bool notifyWindowChanged(PWINDOW_STATE_CHANGE_INFO) override { return true; }

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};
