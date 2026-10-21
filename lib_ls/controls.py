"""Every control, declared once. The UI, presets and the PNG-info round trip
read this table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Control:
    name: str
    label: str
    default: object
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    kind: str = "slider"    # slider | checkbox
    info: str = ""


C = Control
CONTROLS = [
    C("strength", "Overall strength", 1.0, info="0 = original, 1 = the full retouch."),
    # finding the skin
    C("delicate", "Include delicate skin", True, kind="checkbox",
      info="Lips, nipples, intimate areas: retouched lightly, their colour kept."),
    C("tolerance", "Skin match", 1.0, 0.5, 2.0, 0.05,
      info="Higher finds more skin (and more skin-coloured cloth)."),
    C("show_mask", "Show skin mask", False, kind="checkbox",
      info="Adds a picture of what was found: red skin, yellow delicate, blue brows and lashes."),
    # tone
    C("even_tone", "Even skin", 0.3, info="Spots, blotches and uneven patches. Pores are kept."),
    C("even_colour", "Even colour", 0.2, info="Redness and colour patches towards the skin's own colour."),
    C("wrinkles", "Soften lines", 0.10, info="Creases and wrinkles, without flattening them away."),
    # texture
    C("pores", "Pores", 1.0, 0.0, 2.0, 0.05, info="The skin's own fine texture: 1 = as it is, more = crisper."),
    C("add_texture", "Add texture", 0.0, info="Pores and fine relief on plastic-smooth generated skin."),
    C("texture_scale", "Texture size", 1.0, 0.5, 2.0, 0.05, info="Size of the added pores."),
    C("pigmentation", "Natural pigmentation", 0.0, info="Faint uneven melanin, redness and the odd freckle."),
    # finish
    C("glow", "Glow", 0.0, info="A soft sheen on the lit skin."),
    C("eye_detail", "Brows & lashes", 0.15, info="Crisper, a touch deeper. Needs a face."),
]

BY_NAME = {c.name: c for c in CONTROLS}
NAMES = [c.name for c in CONTROLS]
DEFAULTS = {c.name: c.default for c in CONTROLS}

INFOTEXT_KEY = "Living Skin"


def coerce(name, value):
    c = BY_NAME[name]
    if c.kind == "checkbox":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return min(max(float(value), c.minimum), c.maximum)


def settings(values=None, **overrides):
    s = dict(DEFAULTS)
    for src in (values or {}), overrides:
        for k, v in src.items():
            if k in BY_NAME:
                s[k] = coerce(k, v)
    return s


def to_infotext(s) -> str:
    """Only what differs from the defaults: 'on' for a stock run."""
    parts = []
    for c in CONTROLS:
        if c.name == "show_mask":
            continue
        v = s[c.name]
        if v == c.default or (isinstance(v, float) and abs(v - c.default) < 1e-9):
            continue
        parts.append(f"{c.name}={v:g}" if isinstance(v, float) else f"{c.name}={v}")
    return "; ".join(parts) or "on"


def from_infotext(text):
    if not text:
        return None
    values = {}
    for part in text.strip().strip('"').split(";"):
        if "=" in part:
            k, v = (t.strip() for t in part.split("=", 1))
            if k in BY_NAME:
                try:
                    values[k] = coerce(k, v)
                except ValueError:
                    pass
    return settings(values)
