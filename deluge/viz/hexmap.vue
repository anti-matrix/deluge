<template>
  <div class="deluge-hexmap" :style="{position: 'relative', width: '100%', height: height_px + 'px'}">
    <canvas ref="canvas" :style="{width: '100%', height: height_px + 'px', cursor: dragging ? 'grabbing' : 'crosshair',
            background: background, display: 'block'}"></canvas>
    <div v-if="tip_visible && tooltip" :style="tipStyle" v-html="tooltip"></div>
    <div style="position:absolute; right:6px; top:6px; display:flex; gap:4px">
      <button @click="zoomBy(1.5)" style="width:28px;height:28px;background:#fff;color:#111;border:1px solid #888">+</button>
      <button @click="zoomBy(1 / 1.5)" style="width:28px;height:28px;background:#fff;color:#111;border:1px solid #888">&minus;</button>
      <button @click="resetView()" style="height:28px;padding:0 6px;background:#fff;color:#111;border:1px solid #888">reset</button>
    </div>
  </div>
</template>
<script>
module.exports = {
  data() {
    return {scale: 1, ox: 0, oy: 0, dragging: false, moved: false, lastX: 0, lastY: 0,
            tip_visible: false, tipX: 0, tipY: 0, hoverCell: -1, hoverTimer: null};
  },
  computed: {
    tipStyle() {
      return {position: 'absolute', left: (this.tipX + 14) + 'px', top: (this.tipY + 14) + 'px',
              background: 'rgba(255,255,255,0.95)', color: '#111', border: '1px solid #666', padding: '4px 6px',
              'font-size': '12px', 'pointer-events': 'none', 'white-space': 'nowrap', 'z-index': 10};
    },
  },
  mounted() {
    const c = this.$refs.canvas;
    this.ctx = c.getContext('2d');
    this.buildGeometry();
    this.buildFill();
    this.buildLines();
    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(c);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      this.zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.2 : 1 / 1.2);
    }, {passive: false});
    c.addEventListener('mousedown', (e) => {
      this.dragging = true; this.moved = false; this.lastX = e.clientX; this.lastY = e.clientY;
    });
    window.addEventListener('mouseup', this.onUp = (e) => {
      if (this.dragging && !this.moved) {
        const r = c.getBoundingClientRect();
        const cell = this.cellAt(e.clientX - r.left, e.clientY - r.top);
        if (cell >= 0) this.click(cell);
      }
      this.dragging = false;
    });
    c.addEventListener('mousemove', (e) => {
      const r = c.getBoundingClientRect();
      if (this.dragging) {
        const dx = e.clientX - this.lastX, dy = e.clientY - this.lastY;
        if (Math.abs(dx) + Math.abs(dy) > 2) this.moved = true;
        this.ox += dx; this.oy += dy; this.lastX = e.clientX; this.lastY = e.clientY;
        this.tip_visible = false;
        this.draw();
        return;
      }
      this.tipX = e.clientX - r.left; this.tipY = e.clientY - r.top;
      const cell = this.cellAt(this.tipX, this.tipY);
      this.tip_visible = cell >= 0;
      if (cell !== this.hoverCell) {
        this.hoverCell = cell;
        clearTimeout(this.hoverTimer);
        if (cell >= 0) this.hoverTimer = setTimeout(() => this.hover(cell), 40);
      }
    });
    c.addEventListener('mouseleave', () => { this.tip_visible = false; this.hoverCell = -1; });
  },
  beforeDestroy() {
    if (this.resizeObserver) this.resizeObserver.disconnect();
    window.removeEventListener('mouseup', this.onUp);
  },
  beforeUnmount() {
    this.$options.beforeDestroy.call(this);
  },
  watch: {
    colors() { this.buildFill(); this.draw(); },
    lines() { this.buildLines(); this.draw(); },
    marks() { this.draw(); },
    selected() { this.draw(); },
    grid_width() { this.buildGeometry(); this.buildFill(); this.buildLines(); this.resetView(); },
  },
  methods: {
    decode(b64, type) {
      const bin = atob(b64 || '');
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return type === 'f32' ? new Float32Array(bytes.buffer) : bytes;
    },
    // Cell centers (hex units, y up), pointy-top hexes of size 1, even rows shifted right.
    buildGeometry() {
      const W = this.grid_width, H = this.grid_height, s3 = Math.sqrt(3);
      this.cx = new Float32Array(W * H); this.cy = new Float32Array(W * H);
      for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
        const i = x * H + y;
        this.cx[i] = s3 * (x + (y % 2 === 0 ? 0.5 : 0)); this.cy[i] = 1.5 * y;
      }
      this.wx0 = -1; this.wx1 = s3 * (W + 0.5) + 1; this.wy0 = -1.2; this.wy1 = 1.5 * (H - 1) + 1.2;
    },
    hexPath(p, i, k) {
      const x = this.cx[i], y = this.cy[i], h = 0.866 * k, v = k, v2 = 0.5 * k;
      p.moveTo(x, y + v); p.lineTo(x + h, y + v2); p.lineTo(x + h, y - v2);
      p.lineTo(x, y - v); p.lineTo(x - h, y - v2); p.lineTo(x - h, y + v2); p.closePath();
    },
    // One Path2D per colour, so a redraw fills each colour once.
    buildFill() {
      if (!this.cx) return;
      const rgb = this.decode(this.colors), n = this.cx.length, groups = new Map();
      if (rgb.length < 3 * n) { this.fill = []; return; }
      for (let i = 0; i < n; i++) {
        const key = (rgb[3 * i] << 16) | (rgb[3 * i + 1] << 8) | rgb[3 * i + 2];
        let p = groups.get(key);
        if (!p) { p = new Path2D(); groups.set(key, p); }
        this.hexPath(p, i, 1.03);
      }
      this.fill = [...groups.entries()].map(([k, p]) => ['#' + k.toString(16).padStart(6, '0'), p]);
    },
    // Line sets: [{segments (base64 float32 x1 y1 x2 y2 ...), width px, color, dash}], drawn in order.
    buildLines() {
      this.lineSets = (this.lines || []).map((L) => {
        const s = this.decode(L.segments, 'f32'), p = new Path2D();
        for (let k = 0; k + 3 < s.length; k += 4) { p.moveTo(s[k], s[k + 1]); p.lineTo(s[k + 2], s[k + 3]); }
        return {path: p, width: L.width, color: L.color, dash: L.dash || null};
      });
    },
    fit() {
      const c = this.$refs.canvas, w = c.clientWidth, h = c.clientHeight;
      return Math.min(w / (this.wx1 - this.wx0), h / (this.wy1 - this.wy0));
    },
    resetView() { this.scale = 1; this.ox = 0; this.oy = 0; this.draw(); },
    // Screen transform: base fit, then the user's zoom (scale) and pan (ox, oy); y flipped (north up).
    view() {
      const c = this.$refs.canvas, f = this.fit() * this.scale;
      const w = c.clientWidth, h = c.clientHeight;
      const tx = (w - f * (this.wx1 - this.wx0) * 1) / 2 - f * this.wx0 + this.ox;
      const ty = (h + f * (this.wy1 - this.wy0)) / 2 + f * this.wy0 + this.oy;
      return {f, tx, ty};
    },
    toWorld(px, py) { const v = this.view(); return [(px - v.tx) / v.f, (v.ty - py) / v.f]; },
    toScreen(x, y) { const v = this.view(); return [v.tx + v.f * x, v.ty - v.f * y]; },
    zoomAt(px, py, k) {
      const [wx, wy] = this.toWorld(px, py);
      this.scale = Math.min(Math.max(this.scale * k, 1), this.max_zoom);
      const [sx, sy] = this.toScreen(wx, wy);
      this.ox += px - sx; this.oy += py - sy;
      if (this.scale <= 1) { this.ox = 0; this.oy = 0; }  // the whole map: centred again
      this.draw();
    },
    zoomBy(k) { const c = this.$refs.canvas; this.zoomAt(c.clientWidth / 2, c.clientHeight / 2, k); },
    cellAt(px, py) {
      if (!this.cx) return -1;
      const [wx, wy] = this.toWorld(px, py), H = this.grid_height, W = this.grid_width, s3 = Math.sqrt(3);
      let best = -1, bd = 1e9;
      const y0 = Math.round(wy / 1.5);
      for (let y = y0 - 1; y <= y0 + 1; y++) {
        if (y < 0 || y >= H) continue;
        const x0 = Math.round(wx / s3 - (y % 2 === 0 ? 0.5 : 0));
        for (let x = x0 - 1; x <= x0 + 1; x++) {
          if (x < 0 || x >= W) continue;
          const i = x * H + y, d = (this.cx[i] - wx) ** 2 + (this.cy[i] - wy) ** 2;
          if (d < bd) { bd = d; best = i; }
        }
      }
      return bd <= 1.0 ? best : -1;
    },
    resize() {
      const c = this.$refs.canvas, r = window.devicePixelRatio || 1;
      c.width = Math.max(1, c.clientWidth * r); c.height = Math.max(1, c.clientHeight * r);
      this.draw();
    },
    draw() {
      if (!this.ctx || !this.fill) return;
      const ctx = this.ctx, c = this.$refs.canvas, r = window.devicePixelRatio || 1, v = this.view();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.setTransform(r * v.f, 0, 0, -r * v.f, r * v.tx, r * v.ty);
      for (const [color, p] of this.fill) { ctx.fillStyle = color; ctx.fill(p); }
      ctx.lineCap = 'round';
      for (const L of this.lineSets) {
        ctx.strokeStyle = L.color; ctx.lineWidth = L.width / v.f;
        ctx.setLineDash(L.dash ? L.dash.map((d) => d / v.f) : []);
        ctx.stroke(L.path);
      }
      ctx.setLineDash([]);
      const m = this.marks || {};
      const px = (k) => k / v.f;
      for (const [x, y, color] of (m.dots || [])) {   // capitals
        ctx.beginPath(); ctx.arc(x, y, px(m.dot_px || 4), 0, 2 * Math.PI);
        ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = px(1.2); ctx.strokeStyle = '#000'; ctx.stroke();
      }
      for (const [x, y] of (m.squares || [])) {       // monuments
        const s = px(m.square_px || 3);
        ctx.fillStyle = m.square_color || '#000'; ctx.fillRect(x - s, y - s, 2 * s, 2 * s);
      }
      for (const a of (m.arrows || [])) {              // wars: x1 y1 x2 y2 dashed
        const [x1, y1, x2, y2, dashed] = a;
        ctx.strokeStyle = m.arrow_color || '#c00'; ctx.lineWidth = px(m.arrow_px || 2);
        ctx.setLineDash(dashed ? [px(6), px(4)] : []);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.setLineDash([]);
        const ang = Math.atan2(y2 - y1, x2 - x1), hl = px(9);
        ctx.beginPath(); ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - hl * Math.cos(ang - 0.4), y2 - hl * Math.sin(ang - 0.4));
        ctx.lineTo(x2 - hl * Math.cos(ang + 0.4), y2 - hl * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fillStyle = m.arrow_color || '#c00'; ctx.fill();
      }
      const path = m.path || [];
      if (path.length) {                               // the teachers' path and position
        ctx.strokeStyle = m.path_color || '#000'; ctx.lineWidth = px(m.path_px || 2);
        ctx.setLineDash([px(2), px(3)]);
        ctx.beginPath(); ctx.moveTo(path[0][0], path[0][1]);
        for (const [x, y] of path.slice(1)) ctx.lineTo(x, y);
        ctx.stroke(); ctx.setLineDash([]);
      }
      if (m.here) {
        const [x, y] = m.here, s = px(7);
        ctx.beginPath(); ctx.moveTo(x, y + s); ctx.lineTo(x + s, y - s); ctx.lineTo(x - s, y - s); ctx.closePath();
        ctx.fillStyle = m.path_color || '#000'; ctx.fill();
      }
      if (this.selected >= 0 && this.cx) {
        const p = new Path2D(); this.hexPath(p, this.selected, 1.0);
        ctx.strokeStyle = '#ff0'; ctx.lineWidth = px(3); ctx.stroke(p);
      }
    },
  },
};
</script>
