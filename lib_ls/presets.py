"""Retouch presets. Each is the full set of controls (the defaults, then its
changes), so one never inherits leftovers from another. How the skin is
found (delicate skin, skin match, the mask preview) is your own pick and
stays put. They are deliberately light: a retouch should not be noticed."""

from .controls import DEFAULTS

CUSTOM = "Custom"
NOT_IN_PRESETS = {"delicate", "tolerance", "show_mask"}

_P = {
    "Natural": dict(),
    "Clean Even Tone": dict(
        even_tone=0.45, even_colour=0.40, wrinkles=0.10, eye_detail=0.10),
    "Soft Beauty": dict(
        even_tone=0.55, even_colour=0.40, wrinkles=0.30, pores=0.90, glow=0.25, eye_detail=0.25),
    "Editorial Detail": dict(
        even_tone=0.20, even_colour=0.20, wrinkles=0.05, pores=1.25, add_texture=0.10, eye_detail=0.40),
    "Real Skin Restore": dict(
        even_tone=0.05, even_colour=0.10, wrinkles=0.0, pores=1.15, add_texture=0.45, pigmentation=0.25,
        eye_detail=0.15),
    "Mature Skin": dict(
        even_tone=0.25, even_colour=0.25, wrinkles=0.20, pores=1.05, eye_detail=0.15),
    "Body Skin": dict(
        even_tone=0.35, even_colour=0.30, wrinkles=0.10, add_texture=0.20, pigmentation=0.10,
        eye_detail=0.05),
    "Sun-Kissed": dict(
        even_tone=0.25, even_colour=0.10, pores=1.05, add_texture=0.15, pigmentation=0.45, glow=0.30,
        eye_detail=0.15),
}

PRESETS = {name: {**DEFAULTS, **p} for name, p in _P.items()}
CHOICES = [CUSTOM, *PRESETS]

CATEGORIES = {
    "Natural": "Retouch", "Clean Even Tone": "Retouch", "Soft Beauty": "Retouch", "Editorial Detail": "Retouch",
    "Real Skin Restore": "Realism", "Mature Skin": "Realism", "Body Skin": "Realism", "Sun-Kissed": "Realism",
}

DESCRIPTIONS = {
    "Natural": "Blemishes and blotches evened, pores and character kept.",
    "Clean Even Tone": "A cleaner, more even complexion: spots and redness out, texture in.",
    "Soft Beauty": "Beauty retouch: smooth even skin, softened lines, a soft glow; pores eased, not erased.",
    "Editorial Detail": "Crisp real texture and sharp brows and lashes, for close-ups.",
    "Real Skin Restore": "For plastic-smooth generated skin: pores, fine relief and natural pigmentation back.",
    "Mature Skin": "Evens tone and softens lines lightly, keeping the face's age and character.",
    "Body Skin": "Whole-body retouch: even tone, a little added texture and pigmentation, delicate skin kept.",
    "Sun-Kissed": "Freckles and warm uneven pigmentation, with a sheen on the lit skin.",
}
