"""Living Skin - UI and host hook. The work is in lib_ls/."""

import os
import sys
import traceback
import zlib

import gradio as gr
import numpy as np
import torch
from PIL import Image
from modules import devices, script_callbacks, scripts
from modules.ui_components import InputAccordion

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from lib_ls import masks, xyz  # noqa: E402
from lib_ls.carousel import carousel_html, parse_pick  # noqa: E402
from lib_ls.controls import (  # noqa: E402
    BY_NAME, INFOTEXT_KEY, NAMES, coerce, from_infotext, settings, to_infotext,
)
from lib_ls.presets import CATEGORIES, DESCRIPTIONS, NOT_IN_PRESETS, PRESETS  # noqa: E402
from lib_ls.reference import reference_html  # noqa: E402
from lib_ls.retouch import retouch  # noqa: E402

try:
    from modules.paths import models_path
except ImportError:
    models_path = os.path.join(EXTENSION_ROOT, "models")
masks.MODEL_DIR = os.path.join(models_path, "LivingSkin")

GUIDE = ("*Finds every visible skin, face and body, and retouches it the way a retoucher does: blemishes "
         "evened, pores kept or added, delicate skin treated lightly. Runs after Auto Color Corrector, "
         "before Optical Realism.*")

SECTIONS = [
    ("Finding the skin", ["delicate", "tolerance", "show_mask"]),
    ("Tone", ["even_tone", "even_colour", "wrinkles"]),
    ("Texture", ["pores", "add_texture", "texture_scale", "pigmentation"]),
    ("Finish", ["glow", "eye_detail"]),
]

XYZ_ATTR = "_ls_xyz"


def _register_xyz():
    xyz.register("LS", XYZ_ATTR, [
        ("Preset", str, "preset", lambda: list(PRESETS)),
        ("Overall strength", float, "strength", None),
        ("Even skin", float, "even_tone", None),
        ("Even colour", float, "even_colour", None),
        ("Soften lines", float, "wrinkles", None),
        ("Pores", float, "pores", None),
        ("Add texture", float, "add_texture", None),
        ("Natural pigmentation", float, "pigmentation", None),
        ("Glow", float, "glow", None),
        ("Brows & lashes", float, "eye_detail", None),
        ("Skin match", float, "tolerance", None),
    ])


script_callbacks.on_before_ui(_register_xyz)


def _image_seed(p, image):
    """The seed of the image being processed, so added texture repeats."""
    try:
        i = len(p.pixels_after_sampling) - 1
        if 0 <= i < len(p.seeds):
            return int(p.seeds[i])
    except Exception:
        pass
    return zlib.crc32(image.resize((32, 32)).tobytes())


def _mask_preview(image, maps):
    """The picture dimmed, skin in red, delicate skin in yellow, brows and
    lashes in blue."""
    a = np.asarray(image.convert("RGB")).astype(np.float32)
    over = np.stack([maps["skin"], maps["delicate"], maps["features"]], -1) * 255.0
    out = a * 0.35 + over * 0.65
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


class Script(scripts.Script):
    # After correction (Auto Color Corrector, 140), before the camera
    # (Optical Realism, 150) puts grain and blur over the skin.
    sorting_priority = 145

    def title(self):
        return "Living Skin"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        comps = {}

        def add(name):
            c = BY_NAME[name]
            eid = f"ls_{name}_{tab}"
            info = c.info or None
            if c.kind == "checkbox":
                comps[name] = gr.Checkbox(label=c.label, value=c.default, elem_id=eid, info=info)
            else:
                comps[name] = gr.Slider(label=c.label, minimum=c.minimum, maximum=c.maximum, step=c.step,
                                        value=c.default, elem_id=eid, info=info)

        with InputAccordion(False, label="Living Skin", elem_id=f"ls_enabled_{tab}") as enabled:
            gr.Markdown(GUIDE, elem_classes=["ls-guide"])
            gr.HTML(carousel_html(EXTENSION_ROOT, "ls", tab, list(PRESETS), CATEGORIES, DESCRIPTIONS),
                    elem_id=f"ls_preset_car_{tab}")
            pick = gr.Textbox(value="", show_label=False, container=False, elem_id=f"ls_preset_pick_{tab}",
                              elem_classes=["ls-pick"])
            with gr.Row(equal_height=True):
                about = gr.Markdown("*Pick a retouch above, or Reset for the defaults.*",
                                    elem_id=f"ls_preset_about_{tab}")
                reset = gr.Button("Reset", scale=0, min_width=100, elem_id=f"ls_reset_{tab}")
            gr.HTML(reference_html(EXTENSION_ROOT, "ls-ref", "Living Skin before and after"),
                    elem_id=f"ls_ref_{tab}")
            add("strength")
            for title, names in SECTIONS:
                gr.Markdown(f"**{title}**", elem_classes=["ls-section"])
                for n in names:
                    add(n)

        outputs = [comps[n] for n in NAMES]

        def apply_preset(value):
            name = parse_pick(value)
            s = PRESETS.get(name)
            if not s:
                return [gr.update() for _ in range(len(NAMES) + 1)]
            return ([gr.update(value=f"**{name}** · *{DESCRIPTIONS.get(name, '')}*")]
                    + [gr.update() if n in NOT_IN_PRESETS else gr.update(value=s[n]) for n in NAMES])

        pick.change(apply_preset, [pick], [about] + outputs)
        reset.click(lambda: [gr.update(value=""), gr.update(value="*Defaults.*")]
                    + [gr.update(value=v) for v in settings().values()],
                    [], [pick, about] + outputs)

        def field(name):
            def get(params):
                s = from_infotext(params.get(INFOTEXT_KEY, ""))
                return None if s is None else s[name]
            return get

        self.infotext_fields = [(enabled, lambda d: INFOTEXT_KEY in d)]
        self.infotext_fields += [(comps[n], field(n)) for n in NAMES if n != "show_mask"]
        self.paste_field_names = [INFOTEXT_KEY]

        return [enabled, *outputs]

    # After the composite, so an "only masked" inpaint is retouched as a whole
    # image instead of leaving a seam at the crop edge.
    def postprocess_image_after_composite(self, p, pp, enabled, *values):
        axis = xyz.overrides(p, XYZ_ATTR)
        if not (enabled or axis) or pp.image is None:
            return
        s = settings(dict(zip(NAMES, values)))
        if axis:
            s = xyz.merged(s, axis, coerce, BY_NAME, presets=PRESETS, not_in_presets=NOT_IN_PRESETS)
        if s["strength"] <= 0:
            return
        try:
            device = devices.device
            image = pp.image.convert("RGB")
            maps = masks.skin_maps(image, device, tolerance=s["tolerance"], delicate=s["delicate"])
            print(f"[Living Skin] {maps['faces']} face(s), skin {maps['skin'].mean():.0%} of the frame "
                  f"(colour from {maps['source']})")
            x = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            t = lambda a: torch.from_numpy(a)[None, None].to(device)  # noqa: E731
            with torch.no_grad():
                y = retouch(x, t(maps["skin"]), t(maps["delicate"]), t(maps["features"]), s,
                            seed=_image_seed(p, image), scale=maps["scale"])
            out = Image.fromarray((y[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round()
                                  .clip(0, 255).astype(np.uint8))
        except Exception as exc:
            traceback.print_exc()
            print(f"[Living Skin] Error, image left untouched: {exc}")
            return
        if pp.image.mode == "RGBA":
            out.putalpha(pp.image.getchannel("A"))
        pp.image = out
        if s["show_mask"]:
            # A diagnostic rides next to the image; it never replaces it.
            p.extra_result_images.append(_mask_preview(image, maps))
        p.extra_generation_params[INFOTEXT_KEY] = to_infotext(s)

    def postprocess(self, p, processed, *args):
        masks.park()
        devices.torch_gc()
