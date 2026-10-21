"""Before / after reference picture, folded in the panel.

A plain <details> block: it opens and closes in the browser with no server
round trip, and behaves the same on Gradio 3 and 4.
"""

import html
import os

FILE = "reference.jpg"

_ICON_SHOW = ('<svg class="{p}-show" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
              '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>'
              '<rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>')
_ICON_HIDE = ('<svg class="{p}-hide" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
              '<path d="M4 14h6v6"/><path d="M20 10h-6V4"/><path d="M14 10l7-7"/><path d="M3 21l7-7"/></svg>')


def _url(path):
    try:
        from modules.ui_gradio_extensions import webpath
        return webpath(path)
    except Exception:
        return f"file={path.replace(os.sep, '/')}?{int(os.path.getmtime(path))}"


def reference_html(root, prefix, alt, file=FILE, label="Before / after reference"):
    """HTML for the folded reference, or "" if the picture is missing."""
    path = os.path.join(root, file)
    if not os.path.isfile(path):
        return ""
    url = html.escape(_url(path), quote=True)
    p = prefix
    return (
        f'<details class="{p}">'
        f'<summary title="Show or hide the {html.escape(label.lower())}">'
        f'{_ICON_SHOW.format(p=p)}{_ICON_HIDE.format(p=p)}<span>{html.escape(label)}</span></summary>'
        f'<div class="{p}-body">'
        f'<a href="{url}" target="_blank" rel="noopener" title="Open full size in a new tab">'
        f'<img src="{url}" alt="{html.escape(alt, quote=True)}" loading="lazy"></a>'
        f'</div>'
        f'<div class="{p}-hint">Scroll inside the box; click the picture to open it full size.</div>'
        f'</details>'
    )
