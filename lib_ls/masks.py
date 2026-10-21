"""Where the skin is: every visible skin, face and body.

Three small facexlib models (about 80 MB together, downloaded on first use
into models/LivingSkin):

- RetinaFace (mobile) finds the faces;
- BiSeNet face parsing splits each face into skin, brows, eyes, lips, hair...;
- MODNet separates the person from the background.

The face skin then teaches a colour model of this person's skin, which finds
the same skin everywhere else on the body (neck, shoulders, chest, arms,
legs, intimate areas), inside the person matte. A second, pinker and darker
model finds the delicate skin (lips, nipples, genitals, thin skin) that is
retouched more lightly. Without a face, a generic skin-colour prior seeds the
model instead. Everything is worked out at <= 1024 px and returned at full
size as soft 0..1 maps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_models = {}
MODEL_DIR = None            # set by the script: <models>/LivingSkin

# BiSeNet labels (CelebAMask-HQ order).
SKIN, L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, L_EAR, R_EAR, EARRING, NOSE, MOUTH, U_LIP, L_LIP, NECK, \
    NECKLACE, CLOTH, HAIR, HAT = range(1, 19)
FACE_SKIN = (SKIN, L_EAR, R_EAR, NOSE, NECK)
BROWS_EYES = (L_BROW, R_BROW, L_EYE, R_EYE)
NOT_SKIN = (L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, EARRING, MOUTH, NECKLACE, CLOTH, HAIR, HAT)
LIPS = (U_LIP, L_LIP)


def _load(name, device):
    key = (name, str(device))
    if key not in _models:
        for k in [k for k in _models if k[0] == name]:      # one copy per model
            del _models[k]
        if name == "detect":
            from facexlib.detection import init_detection_model
            m = init_detection_model("retinaface_mobile0.25", device=device, model_rootpath=MODEL_DIR)
        elif name == "parse":
            from facexlib.parsing import init_parsing_model
            m = init_parsing_model("bisenet", device=device, model_rootpath=MODEL_DIR)
        else:
            from facexlib.matting import init_matting_model
            m = init_matting_model("modnet", device=device, model_rootpath=MODEL_DIR)
        _models[key] = m
    return _models[key]


def park():
    """Free the models between jobs (they are small; reloading is quick)."""
    _models.clear()


# ------------------------------------------------------------------ helpers

def _resize(a, h, w, mode="bilinear"):
    t = torch.as_tensor(a, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(h, w), mode=mode, align_corners=False)[0, 0].numpy()


def _box(a, r):
    """Mean over a (2r+1)^2 window, via an integral image."""
    if r < 1:
        return a
    p = np.pad(a, r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s[: a.shape[0], : a.shape[1]] / (k * k)


def guided(guide, src, r, eps=1e-3):
    """Edge-aware smoothing of src following guide (He et al.)."""
    mi, ms = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mi * ms
    var = _box(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = ms - a * mi
    return _box(a, r) * guide + _box(b, r)


def _dilate(m, r):
    t = torch.as_tensor(m, dtype=torch.float32)[None, None]
    return F.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[0, 0].numpy()


def to_lab(rgb):
    """sRGB 0..1 (H, W, 3) -> CIE Lab (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _features(lab):
    """Colour features that shading changes little: a and b over lightness."""
    L = np.maximum(lab[..., 0], 12.0)
    return np.stack([lab[..., 1] / L * 50, lab[..., 2] / L * 50], -1), lab[..., 0]


def _prior(rgb):
    """Generic skin likelihood (YCrCb box, soft), for when there is no face."""
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cr = (rgb[..., 0] - y) * 0.713 + 0.5
    cb = (rgb[..., 2] - y) * 0.564 + 0.5
    s = np.clip(1 - np.abs(cr - 0.60) / 0.09, 0, 1) * np.clip(1 - np.abs(cb - 0.42) / 0.10, 0, 1)
    return s * np.clip((y - 0.08) / 0.1, 0, 1)


class _Gauss:
    def __init__(self, f, w=None, shift=(0.0, 0.0), widen=1.0):
        w = np.ones(len(f)) if w is None else w
        w = w / w.sum()
        self.mu = (f * w[:, None]).sum(0) + np.array(shift)
        d = f - (f * w[:, None]).sum(0)
        cov = (d * w[:, None]).T @ d + np.eye(2) * 4.0
        self.icov = np.linalg.inv(cov * widen ** 2)

    def __call__(self, f, tol):
        d = f - self.mu
        m2 = np.einsum("...i,ij,...j->...", d, self.icov, d)
        return np.exp(-0.5 * m2 / (tol * tol))


# ------------------------------------------------------------------ models

@torch.no_grad()
def _faces(rgb8, device):
    det = _load("detect", device)
    bgr = np.ascontiguousarray(rgb8[..., ::-1])
    out = det.detect_faces(bgr, 0.8)
    faces = []
    for row in out if out is not None else []:
        x1, y1, x2, y2 = (float(v) for v in row[:4])
        if min(x2 - x1, y2 - y1) >= 24:
            faces.append((x1, y1, x2, y2))
    return faces


@torch.no_grad()
def _parse(rgb, box, device):
    """Label map (H, W) of one face, 0 outside its crop."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.7
    l, t = int(max(0, cx - s / 2)), int(max(0, cy - s / 2 - 0.05 * s))
    r, b = int(min(w, cx + s / 2)), int(min(h, cy + s / 2 + 0.05 * s))
    crop = torch.from_numpy(rgb[t:b, l:r]).permute(2, 0, 1)[None].float().to(device)
    x = F.interpolate(crop, size=(512, 512), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    logits = _load("parse", device)((x - mean) / std)[0]
    logits = F.interpolate(logits, size=(b - t, r - l), mode="bilinear", align_corners=False)
    lab = np.zeros((h, w), np.int64)
    lab[t:b, l:r] = logits.argmax(1)[0].cpu().numpy()
    return lab


@torch.no_grad()
def _matte(rgb, device):
    h, w = rgb.shape[:2]
    s = 512 / max(h, w)
    rh, rw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device)
    x = F.interpolate((x - 0.5) / 0.5, size=(rh, rw), mode="area")
    _, _, matte = _load("matte", device)(x, True)
    return F.interpolate(matte, size=(h, w), mode="bilinear", align_corners=False)[0, 0].clamp(0, 1).cpu().numpy()


# ------------------------------------------------------------------ the map

def skin_maps(image, device="cpu", tolerance=1.0, delicate=True):
    """image: PIL RGB. Returns dict of full-size float32 maps (H, W):
    skin (all skin, 0..1), delicate (the delicate share of it), features
    (brows, eyes and lashes, for the eye detail), plus 'faces' (count) and
    'source' (what seeded the colour model)."""
    W, H = image.size
    s = min(1.0, 1024 / max(W, H))
    w, h = max(1, int(W * s)), max(1, int(H * s))
    small = image.convert("RGB").resize((w, h)) if s < 1 else image.convert("RGB")
    rgb8 = np.asarray(small)
    rgb = rgb8.astype(np.float32) / 255.0

    try:
        person = _matte(rgb, device)
    except Exception as exc:          # noqa: BLE001 - a missing matte only widens the search
        print(f"[Living Skin] Person matte unavailable ({exc}); using colour alone.")
        person = np.ones((h, w), np.float32)
    if person.mean() < 0.01:
        person = np.ones((h, w), np.float32)

    face_skin = np.zeros((h, w), bool)
    excluded = np.zeros((h, w), np.float32)
    lips = np.zeros((h, w), np.float32)
    feats = np.zeros((h, w), np.float32)
    faces = []
    try:
        faces = _faces(rgb8, device)
    except Exception as exc:          # noqa: BLE001
        print(f"[Living Skin] Face detection unavailable ({exc}).")
    for box in faces:
        lab_map = _parse(rgb, box, device)
        face_skin |= np.isin(lab_map, FACE_SKIN)
        excluded = np.maximum(excluded, np.isin(lab_map, NOT_SKIN).astype(np.float32))
        lips = np.maximum(lips, np.isin(lab_map, LIPS).astype(np.float32))
        eyes = np.isin(lab_map, BROWS_EYES).astype(np.float32)
        # Lashes sit just outside the parsed eye: grow it by ~4% of the face.
        r = max(1, int(0.04 * (box[2] - box[0])))
        feats = np.maximum(feats, _dilate(eyes, r))

    lab = to_lab(rgb)
    f, L = _features(lab)
    seed = face_skin & (excluded < 0.5)
    source = "face"
    if seed.sum() < 300:
        prior = _prior(rgb) * person
        seed = prior > 0.6
        source = "colour prior"
    if seed.sum() < 300:
        skin = _prior(rgb) * person
        dmap = np.zeros_like(skin)
    else:
        main = _Gauss(f[seed])
        p_main = main(f, 2.0 * tolerance)
        # Skin always has some colour: white, grey and black cloth, grey
        # hair and beards do not, whatever their lightness.
        chroma = np.hypot(f[..., 0], f[..., 1])
        c_seed = float(np.median(chroma[seed]))
        colourful = np.clip((chroma - 0.35 * c_seed) / (0.25 * c_seed + 1e-6), 0, 1)
        # Lightness: skin in shadow or highlight still counts, far off does not.
        Lm, Ls = float(L[seed].mean()), float(L[seed].std()) + 4.0
        bright = np.clip(1.0 - (np.abs(L - Lm) - 2.0 * Ls) / (15.0 * tolerance), 0, 1)
        p_main *= colourful * bright
        if delicate:
            # Thin and intimate skin: redder (more a*), a little less yellow,
            # often darker, and more varied.
            dmodel = _Gauss(f[seed], shift=(8.0, -3.0), widen=1.4)
            dark_ok = np.clip(1.0 - (np.abs(L - (Lm - 8.0)) - 2.5 * Ls) / (15.0 * tolerance), 0, 1)
            p_del = dmodel(f, 2.0 * tolerance) * colourful * dark_ok
            # Lips, nipples and intimate skin always sit inside ordinary skin:
            # keep the delicate model near it, so dark red clothes stay out.
            r_near = max(2, int(0.03 * max(w, h)))
            near = np.clip(_box(_dilate((p_main > 0.5).astype(np.float32), r_near), r_near), 0, 1)
            p_del = np.maximum(p_del * (1 - p_main) * near, lips)
        else:
            p_del = np.zeros_like(p_main)
        # Knit, hair, beards and lace are far rougher than skin at fine scale.
        y = lab[..., 0] / 100.0
        rough = _box(np.abs(y - _box(y, 1)), 2)
        r_seed = float(np.median(rough[seed])) + 1e-3
        smooth = np.clip(1.0 - (rough / r_seed - 2.2) / 1.5, 0, 1)
        p_main, p_del = p_main * smooth, np.maximum(p_del * smooth, lips)
        skin = np.maximum(p_main, p_del)
        skin[face_skin & (excluded < 0.5)] = 1.0
        dmap = p_del
    skin = skin * np.clip(person * 1.3, 0, 1) * (1 - excluded)

    # Snap the soft map to the picture's edges, then back to full size.
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    r = max(2, int(0.006 * max(w, h)))
    skin = np.clip(guided(luma, skin.astype(np.float64), r, 2e-3), 0, 1)
    dmap = np.clip(guided(luma, dmap.astype(np.float64), r, 2e-3), 0, 1) * skin

    out = {k: _resize(v, H, W).astype(np.float32) for k, v in
           (("skin", skin), ("delicate", dmap), ("features", feats))}
    out["faces"], out["source"] = len(faces), source
    # The size skin detail is measured against: the faces' width (pores are
    # about 0.7 % of it), or a quarter of the frame when there is no face.
    widths = sorted((b[2] - b[0]) / s for b in faces)
    out["scale"] = float(widths[len(widths) // 2]) if widths else 0.25 * min(W, H)
    out["boxes"] = [tuple(v / s for v in b) for b in faces]
    return out
