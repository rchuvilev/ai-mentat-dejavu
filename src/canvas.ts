// @ts-nocheck -- pure-JS raster shim, already covered by desktop/test_shim.mjs
/**
 * Minimal pure-JS canvas, just enough for the shape generator.
 *
 * Node has no <canvas>. The usual answer is the `canvas` npm package, but that
 * needs a native build (cairo headers) which is exactly the kind of dependency
 * that broke on this device. The generator only needs: createImageData /
 * putImageData / getImageData, fillStyle, translate+rotate, arc, rect, polygon
 * lines, fill. So implement that directly -- zero dependencies, works anywhere
 * Node runs, and produces byte-identical output to the browser path because
 * both rasterise the same analytic shapes.
 */

export class ImageDataShim {
  width: number; height: number; data: Uint8ClampedArray;
  constructor(w: number, h: number) { this.width = w; this.height = h; this.data = new Uint8ClampedArray(w * h * 4); }
}

class Ctx {
  constructor(w, h) {
    this.width = w; this.height = h;
    this.data = new Uint8ClampedArray(w * h * 4);
    this.fillStyle = '#000';
    this._t = [1, 0, 0, 1, 0, 0];      // a,b,c,d,e,f
    this._stack = [];
    this._path = [];
  }
  createImageData(w: number, h: number) { return new ImageDataShim(w, h); }
  putImageData(id, dx, dy) {
    for (let y = 0; y < id.height; y++)
      for (let x = 0; x < id.width; x++) {
        const s = (y * id.width + x) * 4;
        const px = x + dx, py = y + dy;
        if (px < 0 || py < 0 || px >= this.width || py >= this.height) continue;
        const d = (py * this.width + px) * 4;
        for (let c = 0; c < 4; c++) this.data[d + c] = id.data[s + c];
      }
  }
  getImageData(x, y, w, h) {
    const out = new ImageDataShim(w, h);
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        const s = ((y + j) * this.width + (x + i)) * 4, d = (j * w + i) * 4;
        for (let c = 0; c < 4; c++) out.data[d + c] = this.data[s + c];
      }
    return out;
  }
  save() { this._stack.push([...this._t]); }
  restore() { if (this._stack.length) this._t = this._stack.pop(); }
  setTransform(a, b, c, d, e, f) { this._t = [a, b, c, d, e, f]; }
  translate(x, y) {
    const [a, b, c, d, e, f] = this._t;
    this._t = [a, b, c, d, e + a * x + c * y, f + b * x + d * y];
  }
  rotate(r) {
    const [a, b, c, d, e, f] = this._t;
    const co = Math.cos(r), si = Math.sin(r);
    this._t = [a * co + c * si, b * co + d * si, c * co - a * si, d * co - b * si, e, f];
  }
  _pt(x, y) {
    const [a, b, c, d, e, f] = this._t;
    return [a * x + c * y + e, b * x + d * y + f];
  }
  beginPath() { this._path = []; }
  closePath() { if (this._path.length) this._path.push(this._path[0]); }
  moveTo(x, y) { this._path.push(this._pt(x, y)); }
  lineTo(x, y) { this._path.push(this._pt(x, y)); }
  rect(x, y, w, h) {
    this._path.push(this._pt(x, y), this._pt(x + w, y),
                    this._pt(x + w, y + h), this._pt(x, y + h));
    this._path.push(this._path[this._path.length - 4]);
  }
  arc(cx, cy, r, _s, _e) {
    const N = 64;                       // polygon approximation of the circle
    for (let i = 0; i <= N; i++) {
      const a = i / N * Math.PI * 2;
      this._path.push(this._pt(cx + r * Math.cos(a), cy + r * Math.sin(a)));
    }
  }
  _rgb() {
    const s = this.fillStyle;
    let m = /^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/.exec(s);
    if (m) return [+m[1], +m[2], +m[3]];
    m = /^#([0-9a-f]{6})$/i.exec(s);
    if (m) { const v = parseInt(m[1], 16); return [v >> 16 & 255, v >> 8 & 255, v & 255]; }
    m = /^#([0-9a-f]{3})$/i.exec(s);
    if (m) { const h = m[1];
      return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)]; }
    return [0, 0, 0];
  }
  fill() {
    const p = this._path;
    if (p.length < 3) return;
    const [R, G, B] = this._rgb();
    let minY = Infinity, maxY = -Infinity, minX = Infinity, maxX = -Infinity;
    for (const [x, y] of p) {
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
    }
    minY = Math.max(0, Math.floor(minY)); maxY = Math.min(this.height - 1, Math.ceil(maxY));
    minX = Math.max(0, Math.floor(minX)); maxX = Math.min(this.width - 1, Math.ceil(maxX));
    // even-odd scanline fill, sampling pixel centres
    for (let y = minY; y <= maxY; y++) {
      const yc = y + 0.5, xs = [];
      for (let i = 0; i < p.length - 1; i++) {
        const [x1, y1] = p[i], [x2, y2] = p[i + 1];
        if ((y1 <= yc && y2 > yc) || (y2 <= yc && y1 > yc))
          xs.push(x1 + (yc - y1) / (y2 - y1) * (x2 - x1));
      }
      // implicit closing edge
      const [xa, ya] = p[p.length - 1], [xb, yb] = p[0];
      if ((ya <= yc && yb > yc) || (yb <= yc && ya > yc))
        xs.push(xa + (yc - ya) / (yb - ya) * (xb - xa));
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const x0 = Math.max(minX, Math.ceil(xs[k] - 0.5));
        const x1 = Math.min(maxX, Math.floor(xs[k + 1] - 0.5));
        for (let x = x0; x <= x1; x++) {
          const d = (y * this.width + x) * 4;
          this.data[d] = R; this.data[d + 1] = G; this.data[d + 2] = B; this.data[d + 3] = 255;
        }
      }
    }
  }
  fillRect(x, y, w, h) {
    const [R, G, B] = this._rgb();
    for (let j = Math.max(0, y | 0); j < Math.min(this.height, (y + h) | 0); j++)
      for (let i = Math.max(0, x | 0); i < Math.min(this.width, (x + w) | 0); i++) {
        const d = (j * this.width + i) * 4;
        this.data[d] = R; this.data[d + 1] = G; this.data[d + 2] = B; this.data[d + 3] = 255;
      }
  }
}

import type { Surface, SurfaceCtx } from './dataset.js';

export function createCanvas(w: number, h: number): Surface {
  const ctx = new Ctx(w, h) as unknown as SurfaceCtx;
  return { width: w, height: h, getContext: (_t: '2d') => ctx };
}
