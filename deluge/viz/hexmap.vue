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
            tip_visible: false, tipX: 0, tipY: 0, hoverCell: -1, hoverTimer: null, hoverArrow: -1, hoverBanner: -1};
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
      const [kind, k] = this.hitWar(this.tipX, this.tipY);
      const ha = kind === 'arrow' ? k : -1, hb = kind === 'banner' ? k : -1;
      if (ha !== this.hoverArrow || hb !== this.hoverBanner) { this.hoverArrow = ha; this.hoverBanner = hb; this.draw(); }
      const cell = this.cellAt(this.tipX, this.tipY);
      this.tip_visible = cell >= 0;
      if (cell !== this.hoverCell) {
        this.hoverCell = cell;
        clearTimeout(this.hoverTimer);
        if (cell >= 0) this.hoverTimer = setTimeout(() => this.hover(cell), 40);
      }
    });
    c.addEventListener('mouseleave', () => {
      this.tip_visible = false; this.hoverCell = -1;
      if (this.hoverArrow >= 0 || this.hoverBanner >= 0) { this.hoverArrow = -1; this.hoverBanner = -1; this.draw(); }
    });
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
      const lit = this.highlight();                    // linked highlighting (hovered arrow or banner)
      const supply = m.supply || [], arrows = m.arrows || [];
      for (let k = 0; k < supply.length; k++) {         // supply lines: x1 y1 x2 y2 opacity attacker arrow
        const [x1, y1, x2, y2, op] = supply[k], on = lit && lit.supply.has(k);
        ctx.globalAlpha = lit ? (on ? Math.max(op, 0.9) : op * 0.25) : op;
        ctx.strokeStyle = m.supply_color || '#700'; ctx.lineWidth = px((m.supply_px || 1) * (on ? 2.5 : 1));
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
      for (let k = 0; k < arrows.length; k++) {         // campaigns: x1 y1 x2 y2 dashed alpha attacker defender supply
        const [x1, y1, x2, y2, dashed, alpha] = arrows[k], on = lit && lit.arrows.has(k);
        ctx.globalAlpha = lit ? (on ? 1 : alpha * 0.2) : alpha;
        ctx.strokeStyle = m.arrow_color || '#c00'; ctx.lineWidth = px((m.arrow_px || 2) * (on ? 1.8 : 1));
        ctx.setLineDash(dashed ? [px(6), px(4)] : []);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.setLineDash([]);
        const ang = Math.atan2(y2 - y1, x2 - x1), hl = px(on ? 12 : 9);
        ctx.beginPath(); ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - hl * Math.cos(ang - 0.4), y2 - hl * Math.sin(ang - 0.4));
        ctx.lineTo(x2 - hl * Math.cos(ang + 0.4), y2 - hl * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fillStyle = m.arrow_color || '#c00'; ctx.fill();
      }
      ctx.globalAlpha = 1;
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
      this.drawBanners(ctx, r, lit);
    },
    // Linked highlighting: a hovered arrow lights its banners and supply line; a hovered banner
    // lights all its polity's arrows and supply lines.
    highlight() {
      const m = this.marks || {}, arrows = m.arrows || [], banners = m.banners || [];
      if (this.hoverArrow >= 0 && this.hoverArrow < arrows.length) {
        const a = arrows[this.hoverArrow];
        const b = new Set(); banners.forEach((f, k) => { if (f[3] === a[6] || f[3] === a[7]) b.add(k); });
        return {arrows: new Set([this.hoverArrow]), supply: new Set(a[8] >= 0 ? [a[8]] : []), banners: b};
      }
      if (this.hoverBanner >= 0 && this.hoverBanner < banners.length) {
        const pid = banners[this.hoverBanner][3], A = new Set(), S = new Set();
        arrows.forEach((a, k) => { if (a[6] === pid || a[7] === pid) { A.add(k); if (a[8] >= 0) S.add(a[8]); } });
        return {arrows: A, supply: S, banners: new Set([this.hoverBanner])};
      }
      return null;
    },
    // War banners, in screen pixels: a flag on the capital in the lineage colour, a sword with the
    // number of attacks made, a shield with the number suffered.
    drawBanners(ctx, r, lit) {
      const m = this.marks || {}, banners = m.banners || [], s = m.banner_px || 11;
      this.bannerRects = [];
      ctx.setTransform(r, 0, 0, r, 0, 0);
      ctx.font = `bold ${Math.round(0.95 * s)}px sans-serif`; ctx.textBaseline = 'middle';
      for (let k = 0; k < banners.length; k++) {
        const [x, y, color, , made, against] = banners[k];
        const [sx, sy] = this.toScreen(x, y), parts = [];
        if (made > 0) parts.push(['sword', String(made)]);
        if (against > 0) parts.push(['shield', String(against)]);
        let w = 0.3 * s;
        for (const [, n] of parts) w += 0.9 * s + ctx.measureText(n).width + 0.35 * s;
        const h = 1.35 * s, top = sy - 2.3 * s, on = lit && lit.banners.has(k);
        ctx.globalAlpha = lit && !on ? 0.35 : 1;
        ctx.strokeStyle = '#222'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, top); ctx.stroke();                // the pole
        ctx.fillStyle = color; ctx.fillRect(sx, top, w, h);
        ctx.lineWidth = on ? 2.5 : 1; ctx.strokeStyle = on ? '#ff0' : '#222'; ctx.strokeRect(sx, top, w, h);
        let ix = sx + 0.3 * s;
        const cy = top + h / 2;
        for (const [icon, n] of parts) {
          const cx = ix + 0.4 * s;
          ctx.strokeStyle = '#fff'; ctx.fillStyle = '#fff'; ctx.lineWidth = 1.6;
          if (icon === 'sword') {
            ctx.beginPath(); ctx.moveTo(cx, cy - 0.5 * s); ctx.lineTo(cx, cy + 0.25 * s);           // blade
            ctx.moveTo(cx - 0.28 * s, cy + 0.2 * s); ctx.lineTo(cx + 0.28 * s, cy + 0.2 * s);     // guard
            ctx.moveTo(cx, cy + 0.2 * s); ctx.lineTo(cx, cy + 0.5 * s); ctx.stroke();             // grip
          } else {
            ctx.beginPath(); ctx.moveTo(cx - 0.36 * s, cy - 0.42 * s); ctx.lineTo(cx + 0.36 * s, cy - 0.42 * s);
            ctx.lineTo(cx + 0.36 * s, cy); ctx.quadraticCurveTo(cx + 0.3 * s, cy + 0.35 * s, cx, cy + 0.5 * s);
            ctx.quadraticCurveTo(cx - 0.3 * s, cy + 0.35 * s, cx - 0.36 * s, cy); ctx.closePath(); ctx.fill();
          }
          const tx = ix + 0.9 * s;
          ctx.lineWidth = 2.5; ctx.strokeStyle = '#000'; ctx.strokeText(n, tx, cy); ctx.fillText(n, tx, cy);
          ix = tx + ctx.measureText(n).width + 0.35 * s;
        }
        this.bannerRects.push([sx, top, w, h]);
      }
      ctx.globalAlpha = 1;
    },
    // The banner or arrow under the pointer (screen pixels), banners first.
    hitWar(px, py) {
      const m = this.marks || {};
      for (let k = (this.bannerRects || []).length - 1; k >= 0; k--) {
        const [x, y, w, h] = this.bannerRects[k];
        if (px >= x - 2 && px <= x + w + 2 && py >= y - 2 && py <= y + h + 2) return ['banner', k];
      }
      const arrows = m.arrows || [];
      let best = -1, bd = 6;
      for (let k = 0; k < arrows.length; k++) {
        const [x1, y1, x2, y2] = arrows[k], [ax, ay] = this.toScreen(x1, y1), [bx, by] = this.toScreen(x2, y2);
        const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
        const t = L > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L)) : 0;
        const d = Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
        if (d < bd) { bd = d; best = k; }
      }
      return best >= 0 ? ['arrow', best] : [null, -1];
    },
  },
};
</script>
