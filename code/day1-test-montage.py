"""day1-test-montage.py - the test part's map exports and montages for the 252405 letter read.

EXPORTS (full resolution, 2125 x 2265 at 9.612 um, the render grid; unmasked, the mask is render/252405-mask.npy):
  arm-A-R1-tauil_ft6k-rev-mean5.npy / -raw8.png          arm A, the selected recipe R1
  arm-B-best-single-tauil_ft6k-rev-mean5.npy / -raw8.png arm B, the best single member (the same map: frozen
                                                         identical_to_A; written twice so each arm has its file)
  baseline-s43_75k-rev-mean5.npy / -raw8.png             the released s43-75k baseline, same recipe
  depth-null-tauil_ft6k-shuf21-rev-mean5.npy             the arm on the prereg's depth-shuffled input (21-layer slab,
                                                         default_rng(0).permutation(21) before the rev flip), for
                                                         make-pool.py --depth-map
  The .npy is the raw five-centre mean (float32, the 0-255 scale of infer.py's uint8 output), no rescale.
  The -raw8.png is that map rounded and clipped to uint8, nothing else (make-pool.py --display raw shows it as is;
  the card transform clip((p/255 - 0.25)/0.5) is applied only when make-pool reads the .npy).
MONTAGES (same crop in every panel): ScrollPrize 2.4 um map (raw, 0-1) | arm A = arm B | baseline s43-75k, the 9 um
  maps shown through ink_9um's card transform; outside the render mask dark grey. Two crops: the whole frame at
  half scale, and a 1040 px (10.0 mm) square at full resolution about the frame centre. The crop is chosen from the
  frame alone, never from letters.json, and nothing of the reference read is drawn.
Usage: day1-test-montage.py <maps dir> <shuf21 maps dir> <out dir>     CPU only. Run from <workdir>.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
MAPS, SH21, OUT = sys.argv[1:4]
R = 'geom/day1-9um/render/'
C5 = (-4, -2, 0, 2, 4)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def mean5(d, name, order='rev'):
    fs = [os.path.join(d, '%s-c%+d-%s.npy' % (name, c, order)) for c in C5]
    return np.mean([np.load(f).astype(np.float32) for f in fs], 0), {os.path.basename(f): sha256(f) for f in fs}


mask = np.load(R + '252405-mask.npy')
sp = np.load(R + '252405-spmap-grid.npy').astype(np.float32)
arm, arm_src = mean5(MAPS, 'tauil_ft6k')
base, base_src = mean5(MAPS, 's43_75k')
sh21, sh21_src = mean5(SH21, 'tauil_ft6k_shuf21')
log = {'written': time.strftime('%Y-%m-%dT%H:%M:%S'), 'grid': list(mask.shape), 'pitch_um': 9.612, 'mask': R + '252405-mask.npy', 'files': {}}


def export(stem, a, src, png=True):
    p = os.path.join(OUT, stem + '.npy')
    np.save(p, a.astype(np.float32))
    ent = {'npy': p, 'npy_sha256': sha256(p), 'sources': src, 'mean_in_mask': float(a[mask].mean()), 'sd_in_mask': float(a[mask].std())}
    if png:
        q = os.path.join(OUT, stem + '-raw8.png')
        Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8)).save(q)
        ent['png'], ent['png_sha256'] = q, sha256(q)
    log['files'][stem] = ent
    print('wrote %s  sha256 %s' % (p, ent['npy_sha256']))


export('arm-A-R1-tauil_ft6k-rev-mean5', arm, arm_src)
export('arm-B-best-single-tauil_ft6k-rev-mean5', arm, arm_src)
export('baseline-s43_75k-rev-mean5', base, base_src)
export('depth-null-tauil_ft6k-shuf21-rev-mean5', sh21, sh21_src, png=False)
assert log['files']['arm-A-R1-tauil_ft6k-rev-mean5']['npy_sha256'] == log['files']['arm-B-best-single-tauil_ft6k-rev-mean5']['npy_sha256']


def card(p):
    return np.clip((p / 255.0 - 0.25) / 0.5, 0, 1)


def panel(v):
    g = np.where(mask, v, 0.18)
    return (np.clip(g, 0, 1) * 255).astype(np.uint8)


PANELS = [('ScrollPrize 2.4 um map (raw)', panel(np.clip(sp, 0, 1))), ('arm A = arm B: tauil_ft6k R1, rev mean5 (card)', panel(card(arm))),
          ('baseline: released s43-75k, rev mean5 (card)', panel(card(base)))]


def montage(name, ys, xs, scale):
    ims = []
    for title, a in PANELS:
        im = Image.fromarray(a[ys, xs])
        if scale != 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        ims.append((title, im))
    w, h = ims[0][1].size
    gap, top = 12, 28
    out = Image.new('L', (len(ims) * w + (len(ims) - 1) * gap, h + top), 255)
    d = ImageDraw.Draw(out)
    for k, (title, im) in enumerate(ims):
        out.paste(im, (k * (w + gap), top))
        d.text((k * (w + gap) + 4, 8), title, fill=0)
    p = os.path.join(OUT, name)
    out.save(p)
    log['files'][name] = {'png': p, 'png_sha256': sha256(p), 'rows': [ys.start, ys.stop], 'cols': [xs.start, xs.stop], 'scale': scale}
    print('wrote %s' % p)


H, W = mask.shape
montage('montage-full-half.png', slice(0, H), slice(0, W), 0.5)
cy, cx, S = H // 2, W // 2, 1040
montage('montage-centre-10mm-full.png', slice(cy - S // 2, cy + S // 2), slice(cx - S // 2, cx + S // 2), 1)
json.dump(log, open(os.path.join(OUT, 'exports.json'), 'w'), indent=1)
