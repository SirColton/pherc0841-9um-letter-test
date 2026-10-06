"""day1-montage-baselines.py - a 2-column montage of w00big maps on the 9.61 um grid, for the eye.

Panels are named on the command line as TITLE=SPEC, where SPEC is
  sp        ScrollPrize's w00 2.4 um ink map (ds8 jpg, bilinear onto the grid, raw 0-255)
  labels    ScrollPrize's w00 inklabels_v2 on the grid (scoring only, never a model input)
  h24       our 2.4 um Hecate rev map of the w00big render, 4x4 pooled
  <maps dir>/<model>:<order>:<c+0 | mean5 | mean3>   an ink_9um-format map from day1-ink9-maps.py
Model maps are stretched 1st-99th percentile inside the scored frame; the 2.4 maps and labels are
shown raw. A 1 mm bar (104 px) sits bottom-left of every panel. Outside the frame is grey.
CPU only. Usage: day1-montage-baselines.py <out.png> TITLE=SPEC [TITLE=SPEC ...]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates

V = 'geom/l0-1203-render/audit/verify-nine-um-test-does-it-matter'
B = 'geom/l0-1203-render/audit/build-deciding-test'
CTRL24 = 'geom/c0841/mesh0b-render/w00big'
H, W, P, E = 995, 1245, 4, 40
mask = np.load(os.path.join(B, 'ours-mask.npy'))
m24 = np.asarray(Image.open(os.path.join(CTRL24, 'mask.png')), np.float32)[:H * P, :W * P].reshape(H, P, W, P).min((1, 3)) > 0
M = mask & m24; M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False


def load(spec):
    if spec == 'sp':
        ink = np.asarray(Image.open('geom/c0841/ink-20260220213127.jpg').convert('L'), np.float32)
        gi = 2.5 * (380 + (np.arange(H) * P + (P - 1) / 2) / 20.0) - 0.5
        gj = 2.5 * (500 + (np.arange(W) * P + (P - 1) / 2) / 20.0) - 0.5
        GJ, GI = np.meshgrid(gj, gi)
        return map_coordinates(ink, [GI, GJ], order=1), False
    if spec == 'labels':
        return np.load(os.path.join(V, 'w00_inklabels_v2-grid.npy')) * 255, False
    if spec == 'h24':
        return np.asarray(Image.open(os.path.join(CTRL24, 'hecate-rev.png')), np.float32)[:H * P, :W * P].reshape(H, P, W, P).mean((1, 3)), False
    d, mod, o, rec = spec.rsplit('/', 1)[0], *spec.rsplit('/', 1)[1].split(':')
    cs = {'mean5': (-4, -2, 0, 2, 4), 'mean3': (-4, 0, 4)}.get(rec)
    cs = cs or (int(rec[1:]),)
    return np.mean([np.load(os.path.join(d, '%s-c%+d-%s.npy' % (mod, c, o))) for c in cs], 0), True


def panel(a, stretch, title):
    a = np.asarray(a, np.float32)
    if stretch:
        lo, hi = np.percentile(a[M], (1, 99))
        a = (a - lo) / max(hi - lo, 1e-6) * 255
    g = np.clip(a, 0, 255).astype(np.uint8)
    g[~M] = 90
    im = Image.fromarray(g).convert('RGB')
    d = ImageDraw.Draw(im)
    d.rectangle([E, H - E - 14, E + 104, H - E - 6], fill=(255, 140, 0))
    d.text((E, H - E - 44), '1 mm', fill=(255, 140, 0), font=FONT)
    out = Image.new('RGB', (W, H + 40), (20, 20, 20))
    out.paste(im, (0, 40))
    ImageDraw.Draw(out).text((8, 8), title, fill=(240, 240, 240), font=FONT)
    return out


try:
    FONT = ImageFont.truetype('arial.ttf', 22)
except Exception:
    FONT = ImageFont.load_default()
outp, specs = sys.argv[1], [a.split('=', 1) for a in sys.argv[2:]]
pans = [panel(*load(s), t) for t, s in specs]
cols = 2
rows = (len(pans) + cols - 1) // cols
mont = Image.new('RGB', (cols * W + (cols - 1) * 10, rows * (H + 40) + (rows - 1) * 10), (0, 0, 0))
for k, p in enumerate(pans):
    mont.paste(p, ((k % cols) * (W + 10), (k // cols) * (H + 50)))
mont.save(outp)
print('wrote %s %dx%d, %d panels' % (outp, mont.size[0], mont.size[1], len(pans)))
