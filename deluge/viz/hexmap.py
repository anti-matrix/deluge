"""The browser map: a canvas widget that draws the hex grid client-side (``hexmap.vue``).

Python sends, per frame, one RGB colour per cell (base64, flat cell-id order), line sets
(borders: base64 float32 segments in hex units) and marks (capitals, monuments, the
teachers' path, wars as arrows); the browser keeps the hex geometry, so zoom (wheel or
buttons), pan (drag) and hover need no round trip. Hovering a cell calls ``on_hover(cell)``
and shows the ``tooltip`` HTML Python sets back; clicking calls ``on_click(cell)``.
"""

from __future__ import annotations

import base64

import ipyvue
import numpy as np
import traitlets


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


class HexMap(ipyvue.VueTemplate):
    template_file = (__file__, "hexmap.vue")

    grid_width = traitlets.Int(0).tag(sync=True)
    grid_height = traitlets.Int(0).tag(sync=True)
    height_px = traitlets.Int(560).tag(sync=True)
    max_zoom = traitlets.Float(40.0).tag(sync=True)
    background = traitlets.Unicode("#ffffff").tag(sync=True)
    colors = traitlets.Unicode("").tag(sync=True)
    lines = traitlets.List([]).tag(sync=True)
    marks = traitlets.Dict({}).tag(sync=True)
    tooltip = traitlets.Unicode("").tag(sync=True)
    selected = traitlets.Int(-1).tag(sync=True)
    on_hover = traitlets.Callable(None, allow_none=True)
    on_click = traitlets.Callable(None, allow_none=True)

    def vue_hover(self, cell):
        if self.on_hover is not None:
            self.on_hover(int(cell))

    def vue_click(self, cell):
        if self.on_click is not None:
            self.on_click(int(cell))
