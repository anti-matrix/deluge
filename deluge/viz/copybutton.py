"""A button that copies a text to the browser's clipboard (``copybutton.vue``): the bard's
input (``analysis/bard.py``) for pasting into a chat."""

from __future__ import annotations

import ipyvue
import traitlets


class CopyButton(ipyvue.VueTemplate):
    template_file = (__file__, "copybutton.vue")

    text = traitlets.Unicode("").tag(sync=True)
    label = traitlets.Unicode("Copy").tag(sync=True)
    done_label = traitlets.Unicode("Copied").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
