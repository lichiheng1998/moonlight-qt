#version 450
// YCbCr (3-plane) -> RGB for the PyroWave present pass.
//
// Handles both SDR (BT.709 / BT.2020 SDR, 8-bit R8 planes, UNORM swapchain) and
// HDR (BT.2020 PQ, 16-bit R16 planes, HDR10/ST2084 swapchain).
//
// For HDR the captured RGB was PQ-encoded before the encoder's RGB->YCbCr step,
// so inverting the YCbCr matrix yields PQ-encoded RGB again. An HDR10 (ST2084)
// swapchain interprets its values as PQ BT.2020, so we output those RGB values
// directly with no additional transfer applied.
layout(set = 0, binding = 0) uniform texture2D Y;
layout(set = 0, binding = 1) uniform texture2D Cb;
layout(set = 0, binding = 2) uniform texture2D Cr;
layout(set = 0, binding = 3) uniform sampler Samp;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 FragColor;

layout(constant_id = 0) const bool FullRange = false;  // JPEG (full) vs MPEG (limited)
layout(constant_id = 1) const bool BT2020    = false;  // BT.2020 vs BT.709 matrix
layout(constant_id = 2) const bool HdrPQ     = false;  // output PQ BT.2020 for HDR10

// Y'CbCr -> R'G'B' matrices. Column-major: rgb = M * vec3(y, cb, cr)
// with cb,cr already centered at 0.
const mat3 yuv2rgb_bt709 = mat3(
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, -0.13397432 / 0.7152, 1.8556),
    vec3(1.5748, -0.33480248 / 0.7152, 0.0));

// BT.2020 NCL: Kr = 0.2627, Kb = 0.0593, Kg = 0.6780.
//   R = Y + 1.4746 * Cr
//   G = Y - 0.16455313 * Cb - 0.57135313 * Cr
//   B = Y + 1.8814 * Cb
const mat3 yuv2rgb_bt2020 = mat3(
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, -0.16455313, 1.8814),
    vec3(1.4746, -0.57135313, 0.0));

void main()
{
    float y  = textureLod(sampler2D(Y,  Samp), vUV, 0.0).x;
    float cb = textureLod(sampler2D(Cb, Samp), vUV, 0.0).x;
    float cr = textureLod(sampler2D(Cr, Samp), vUV, 0.0).x;

    cb -= 0.5;
    cr -= 0.5;

    if (!FullRange)
    {
        // Limited (MPEG) range expansion. The 16/235 luma and 16/240 chroma
        // footroom/headroom is the same fraction of the value range at any bit
        // depth, so the 8-bit constants apply to normalized samples directly.
        y -= 16.0 / 255.0;
        y *= 255.0 / 219.0;
        const float ChromaScale = 255.0 / 224.0;
        cb *= ChromaScale;
        cr *= ChromaScale;
        y = clamp(y, 0.0, 1.0);
        cb = clamp(cb, -0.5, 0.5);
        cr = clamp(cr, -0.5, 0.5);
    }

    vec3 rgb = (BT2020 ? yuv2rgb_bt2020 : yuv2rgb_bt709) * vec3(y, cb, cr);

    // SDR swapchains are UNORM and HDR10 expects PQ values in [0,1]; clamp the
    // displayable range either way. For HDR (HdrPQ) the RGB is already
    // PQ-encoded BT.2020 and passes through unchanged.
    rgb = clamp(rgb, 0.0, 1.0);

    FragColor = vec4(rgb, 1.0);
}
