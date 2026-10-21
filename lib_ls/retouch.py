"""The retouch itself, on the skin map. Pure torch, (1, 3, H, W) sRGB 0..1.

The skin is split by size into bands, as a retoucher does with frequency
separation, and each band is treated for what lives in it. Sizes are
measured against the face width (or a quarter of the frame without a face),
since pores are a fixed share of a face, not of the picture:

    texture   finer than ~0.7 % of a face: pores, fine lines, peach fuzz.
              Kept (or crisped, or added where the generator smoothed it).
    blemish   up to ~3.5 %: spots, blotches, uneven patches, creases.
              Evened out; dark lines softened on their own.
    tone      up to ~12 %: redness, pigmentation, colour patches.
              Their colour evened towards the skin's own.
    base      the shape and light of the body. Never touched.

Delicate skin (lips, nipples, intimate areas, thin skin) keeps its own colour
and gets only a little added texture.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)


def luma(x):
    w = x.new_tensor(LUMA).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def _gauss_small(x, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    mode = "reflect" if r < min(x.shape[-2:]) else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def blur(x, sigma):
    """Gaussian blur; large ones are done at a lower resolution (same result,
    a fraction of the cost)."""
    if sigma <= 0.3:
        return x
    if sigma <= 6:
        return _gauss_small(x, sigma)
    f = sigma / 3.0
    h, w = x.shape[-2:]
    small = F.interpolate(x, size=(max(1, int(h / f)), max(1, int(w / f))), mode="area")
    small = _gauss_small(small, 3.0)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def _noise(shape, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    return torch.randn(shape, generator=g).to(device, dtype)


def _band_noise(h, w, lo, hi, seed, device, dtype):
    """Noise with features between sigma lo and hi, unit spread."""
    n = _noise((1, 1, h, w), seed, device, dtype)
    b = blur(n, lo) - blur(n, hi)
    return b / (b.std() + 1e-6)


def retouch(x, skin, delicate, features, s, seed=0, scale=None):
    """x (1, 3, H, W); skin, delicate, features (1, 1, H, W) 0..1; s settings;
    scale: face width in px (None: a quarter of the frame)."""
    _, _, h, w = x.shape
    short = min(h, w)
    ref = scale or 0.25 * short
    k = s["texture_scale"]
    s1 = max(0.7, 0.007 * ref)             # texture | blemish
    s2 = max(2.0, 0.035 * ref)             # blemish | tone
    s3 = max(6.0, 0.12 * ref)              # tone | base
    m = skin
    d = delicate.clamp(0, 1) * m
    ordinary = (m - d).clamp(0, 1)

    g1, g2, g3 = blur(x, s1), blur(x, s2), blur(x, s3)
    texture, blemish, tone, base = x - g1, g1 - g2, g2 - g3, g3

    # Every clear edge stays: eyelids, lip lines, nostrils, a jaw, a fold of
    # skin, hair crossing the skin. They are found at the fine scale and
    # judged against this picture's own skin, so the strongest third of the
    # local edges on the skin are kept whole.
    y1 = luma(g1)
    gx = F.pad(y1[..., :, 1:] - y1[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(y1[..., 1:, :] - y1[..., :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(gx * gx + gy * gy)
    on_skin = grad[m > 0.5]
    if on_skin.numel() > 64:
        sample = on_skin[:: max(1, on_skin.numel() // 200000)].float()
        lo, hi = torch.quantile(sample, 0.55), torch.quantile(sample, 0.85)
    else:
        lo, hi = grad.new_tensor(0.01), grad.new_tensor(0.03)
    edge = ((grad - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    r = max(1, int(round(s1)))
    edge = blur(F.max_pool2d(edge, 2 * r + 1, stride=1, padding=r), s1)
    keep_shape = 1.0 - edge

    # Smoothing stays inside the skin, clear of its outline, so the edge of
    # a face against hair or a body against the background keeps its bite.
    rr = max(1, int(round(s2)))
    inner = blur(-F.max_pool2d(-m, 2 * rr + 1, stride=1, padding=rr), s1)

    # -- blemish band: even it; soften dark lines (creases, wrinkles) more.
    even = s["even_tone"] * inner * keep_shape
    bl_y = luma(blemish)
    bl_c = blemish - bl_y
    dark = bl_y.clamp(max=0.0)
    bl_y = bl_y * (1 - even) - dark * (s["wrinkles"] * inner * keep_shape) * (1 - even)
    bl_c = bl_c * (1 - even)
    blemish = bl_y + bl_c

    # -- tone band: blotchy colour (redness, patches) towards the skin's own.
    # Colour only: the light and shade here is the shape of the face.
    # Delicate skin keeps its colour.
    t_y = luma(tone)
    t_c = tone - t_y
    t_c = t_c * (1 - s["even_colour"] * (ordinary * inner).clamp(0, 1))
    tone = t_y + t_c

    # -- texture band: keep, crisp or soften the real pores.
    gain = 1.0 + (s["pores"] - 1.0) * (ordinary * inner + 0.4 * d)
    texture = texture * gain

    out = base + tone + blemish + texture

    # -- added texture: pores as a soft bump map, lit from the top left, so
    # plastic-smooth generated skin gets a surface again.
    if s["add_texture"] > 0:
        pore = max(0.5, 0.004 * ref * k)               # pore radius, px
        hgt = _band_noise(h, w, 0.5 * pore, 1.5 * pore, seed, x.device, x.dtype)
        pits = F.relu(-hgt - 1.0)                       # sparse, round pits
        relief = _band_noise(h, w, 1.5 * pore, 4.0 * pore, seed + 7, x.device, x.dtype)
        field = -pits + 0.25 * relief                   # height: pits below a gently rolling surface
        field = field - blur(field, 6 * pore)
        fx = F.pad(field[..., :, 1:] - field[..., :, :-1], (0, 1, 0, 0))
        fy = F.pad(field[..., 1:, :] - field[..., :-1, :], (0, 0, 0, 1))
        shade = (fx + fy) * (0.9 / pore ** 0.5) + 0.35 * field   # lit from the top left
        y = luma(out).clamp(0.02, 1.0)
        amount = s["add_texture"] * 0.03 * (ordinary + 0.25 * d) * (0.3 + y)
        out = out + shade * amount

    # -- natural pigmentation: faint uneven melanin and redness, and the odd
    # freckle, the variation real skin has and generated skin often lacks.
    if s["pigmentation"] > 0:
        a = s["pigmentation"]
        mel = _band_noise(h, w, 0.5 * s2, 2.0 * s2, seed + 1, x.device, x.dtype)
        red = _band_noise(h, w, 0.8 * s2, 3.0 * s2, seed + 2, x.device, x.dtype)
        frk = _band_noise(h, w, max(0.5, 0.004 * ref * k), max(1.2, 0.012 * ref * k), seed + 3, x.device, x.dtype)
        frk = F.relu(frk - 2.2)
        brown = x.new_tensor([-0.010, -0.018, -0.026]).view(1, 3, 1, 1)
        pink = x.new_tensor([0.010, -0.010, -0.006]).view(1, 3, 1, 1)
        out = out + ordinary * a * (brown * (mel.clamp(min=0) + 1.6 * frk) + pink * red.clamp(min=0))

    # -- glow: a light sheen on the brightest skin, no haze.
    if s["glow"] > 0:
        y = luma(g2)
        lit = ((y - 0.55) / 0.40).clamp(0, 1) ** 2
        out = out + s["glow"] * 0.10 * inner * lit * (1 - out)

    # -- brows and lashes: crisper and a touch deeper, nothing else.
    if s["eye_detail"] > 0:
        f = features
        detail = luma(x - blur(x, max(0.8, 0.004 * ref)))
        out = out + s["eye_detail"] * 1.2 * f * detail
        lc = x - blur(x, max(2.0, 0.02 * ref))
        out = out + s["eye_detail"] * 0.35 * f * lc

    out = out.clamp(0, 1)
    return torch.lerp(x, out, s["strength"])
