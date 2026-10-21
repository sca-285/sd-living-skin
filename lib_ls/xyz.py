"""X/Y/Z plot axes.

The X/Y/Z plot script only sees the controls of an always-on extension if the
extension adds its own axis options to it. Each axis here stores its value on
the cell's processing object (p is copied per cell), and the post-processing
hook lays those values over the panel's settings. A cell that sets any of
this extension's axes switches the extension on for that cell, and an axis
that belongs to a section ticks that section, so the axis always shows.
"""

from __future__ import annotations


def _xyz_module():
    """The loaded xyz_grid script module, or None (older hosts call it xy_grid)."""
    try:
        from modules import scripts
    except ImportError:
        return None
    for data in getattr(scripts, "scripts_data", []):
        if data.script_class.__module__ in ("xyz_grid.py", "xy_grid.py") and hasattr(data, "module"):
            return data.module
    return None


def _setter(attr, key):
    def apply(p, x, xs):
        # A fresh dict: p is a shallow copy per cell, so mutating a shared
        # dict in place would leak one cell's values into the next.
        values = dict(getattr(p, attr, None) or {})
        values[key] = x
        setattr(p, attr, values)
    return apply


def overrides(p, attr):
    """This cell's axis values, {} outside an X/Y/Z run."""
    return getattr(p, attr, None) or {}


def register(prefix, attr, axes):
    """Add axes to the X/Y/Z plot. axes: (label, type, key, choices or None).
    Safe to call more than once: labels already present are skipped."""
    xyz = _xyz_module()
    if xyz is None:
        return False
    have = {o.label for o in xyz.axis_options}
    for label, kind, key, choices in axes:
        full = f"[{prefix}] {label}"
        if full in have:
            continue
        kwargs = {"choices": choices} if choices is not None else {}
        xyz.axis_options.append(xyz.AxisOption(full, kind, _setter(attr, key), **kwargs))
    return True


def merged(s, values, coerce, by_name, group_of=None, presets=None, not_in_presets=()):
    """Settings s with one cell's axis values laid over them.

    A 'preset' value is applied first, the way the preset picker does (what
    presets never touch keeps the panel's value); then every other value, each
    ticking the section it belongs to.
    """
    s = dict(s)
    name = values.get("preset")
    if presets and name in presets:
        for k, v in presets[name].items():
            if k not in not_in_presets:
                s[k] = v
    for k, v in values.items():
        if k == "preset" or k not in by_name:
            continue
        s[k] = coerce(k, v)
        group = (group_of or {}).get(k)
        if group:
            s[group] = True
    return s


def bools():
    return ["True", "False"]
