"""student-test-valstrip.py - P4 of prereg-student.json: the w00 validation-strip sanity table. Sanity only; it chooses
nothing (no step, lr, arm or checkpoint), and it touches no 252405 file.

  crop   CPU. Cuts the w00 full render (renders/w00-stack.npy, the stack the arms trained on) around the strip with a
         256 px margin, so day 1's frozen inference (student-test-ink9-maps.py --stack) can run on it. Writes
         <out>/w00-valstrip-crop-stack.npy and <out>/crop.json (crop box, sha256s).
  score  CPU. Five-centre rev mean of base TAUIL ft6k, the student and the scramble twin on that crop (maps in
         <out>/maps), scored on the strip only:
           AUCsup on S3 20260918 w00 labels by index (renders/w00-inklabels-grid.npy, -supervision-grid.npy):
             pos label >= 0.75; neg label <= 0 AND supervision >= 0.75; midrank AUC.
           hp r at 192 um, blur 2 (masked on the render mask), key never blurred, against ScrollPrize's map
             (renders/w00-spmap-grid.npy) and against Hecate 2.4 rev (geom/c0841/mesh0b-render/w00big/hecate-rev.png,
             4 x 4 block mean by index, as day1-render-mapalign.py reads it, at w00 render px (1900 + i, 2500 + j)).
             m = strip px AND render mask AND Hecate's mask (so both keys use the same px) [machine]; hp and core as
             day1-score-baselines.py: hp(x) = x - G(x m)/max(G(m), 1e-3), core = m AND G(m, 2 sigma) > 0.999.
             No null is computed: the strip is 250 px wide, narrower than day 1's 312 px shifts.
Usage: student-test-valstrip.py crop <out dir> | score <out dir>      Run from <workdir>.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

import runfile

MODE, OUT = sys.argv[1], sys.argv[2]
REN = 'geom/student-9um/renders/'
FREEZE = 'geom/student-9um/freeze/geom/'
PRE = 'geom/student-9um/prereg-student.json'
PRE_SHA = 'f3ebf0abc3f93a6772ce31f6915d7f5aef2ea4afc72badeb11cbb735216d3710'
MARGIN = 256
PX = 2.403 * 4
CKPTS = {'base': ('geom/day1-9um/select/ckpt/tauil_ft6k.pth', 'e3bc6fdfa0c0a32e0a75713ea0e469db14c7305403a4aa1a9b17bff3bf30ef58'),
         'student': ('geom/student-9um/train-student/student-final-step012000.pth', '2a1e01914a2615812a689238e9c92db3484e44f6b7d5f97eac93920041ff8dc0'),
         'scramble': ('geom/student-9um/train-scramble/scramble-final-step012000.pth', 'e5de283853dcdca95dd5fb3ca0963f0793a790615ad39ff34976c7ca63ae02a1')}
os.makedirs(OUT, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


assert sha256(PRE) == PRE_SHA, 'prereg sha256 differs'
pre = json.load(open(PRE))
vs_sha = pre['2_training_regions']['masks']['files_sha256']['geom/student-9um/freeze/geom/w00-valstrip.npy']
assert sha256(FREEZE + 'w00-valstrip.npy') == vs_sha, 'w00-valstrip.npy sha256 differs from the prereg'
mask_full = np.load(REN + 'w00-mask.npy')
H, W = mask_full.shape
cells = np.load(FREEZE + 'w00-valstrip.npy')
strip_full = cells[np.ix_(np.arange(H) // 5, np.arange(W) // 5)] & mask_full
ys, xs = np.nonzero(strip_full)
sy0, sy1, sx0, sx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
y0, y1 = max(0, sy0 - MARGIN), min(H, sy1 + MARGIN)
x0, x1 = max(0, sx0 - MARGIN), min(W, sx1 + MARGIN)

if MODE == 'crop':
    out_txt = os.path.join(OUT, 'crop-run.txt')
    runfile.claim(out_txt)
    say('student-test-valstrip.py crop  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    say('  prereg sha256 %s (matches); w00-valstrip.npy sha256 %s (matches the prereg)' % (PRE_SHA, vs_sha))
    st = np.load(REN + 'w00-stack.npy', mmap_mode='r')
    crop = np.ascontiguousarray(st[:, y0:y1, x0:x1])
    dst = os.path.join(OUT, 'w00-valstrip-crop-stack.npy')
    np.save(dst, crop)
    info = {'written': time.strftime('%Y-%m-%dT%H:%M:%S'), 'source': REN + 'w00-stack.npy', 'source_sha256': sha256(REN + 'w00-stack.npy'),
            'strip_px_full_render': int(strip_full.sum()), 'strip_bbox_full_render': [sy0, sy1, sx0, sx1], 'margin_px': MARGIN,
            'crop_full_render_rows_cols': [y0, y1, x0, x1], 'crop_shape': list(crop.shape), 'output': dst, 'output_sha256': sha256(dst)}
    json.dump(info, open(os.path.join(OUT, 'crop.json'), 'w'), indent=1)
    say(json.dumps(info, indent=1))
    runfile.done(out_txt, '\n'.join(lines) + '\n')
    sys.exit(0)

assert MODE == 'score'
out_txt = os.path.join(OUT, 'valstrip-score.txt')
runfile.claim(out_txt)
say('student-test-valstrip.py score  %s  (P4 sanity: logged only, chooses nothing)' % time.strftime('%Y-%m-%d %H:%M:%S'))
for n, (p, want) in CKPTS.items():
    got = sha256(p)
    say('  %-8s %s sha256 %s (%s)' % (n, p, got, 'matches' if got == want else 'MISMATCH'))
    assert got == want
mask = mask_full[y0:y1, x0:x1]
strip = strip_full[y0:y1, x0:x1]
lab = np.load(REN + 'w00-inklabels-grid.npy', mmap_mode='r')[y0:y1, x0:x1].astype(np.float32)
sup = np.load(REN + 'w00-supervision-grid.npy', mmap_mode='r')[y0:y1, x0:x1].astype(np.float32)
sp = np.load(REN + 'w00-spmap-grid.npy', mmap_mode='r')[y0:y1, x0:x1].astype(np.float32) * 255
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
BH, BW, P, BR, BC = 995, 1245, 4, 1900, 2500
h24b = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/hecate-rev.png'), np.float32)[:BH * P, :BW * P].reshape(BH, P, BW, P).mean((1, 3))
m24b = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/mask.png'), np.float32)[:BH * P, :BW * P].reshape(BH, P, BW, P).min((1, 3)) > 0
h24 = np.zeros((H, W), np.float32); m24 = np.zeros((H, W), bool)
h24[BR:BR + BH, BC:BC + BW] = h24b; m24[BR:BR + BH, BC:BC + BW] = m24b
h24, m24 = h24[y0:y1, x0:x1], m24[y0:y1, x0:x1]
pos = strip & (lab >= 0.75)
neg = strip & (lab <= 0) & (sup >= 0.75)
m = strip & mask & m24
say('  crop rows [%d, %d) cols [%d, %d) of the w00 render; strip %d px (bbox rows [%d, %d) cols [%d, %d)); S3 20260918 pos %d neg(sup) %d;'
    ' hp m (strip AND Hecate mask) %d px, strip px outside Hecate mask %d' % (y0, y1, x0, x1, strip.sum(), sy0, sy1, sx0, sx1, pos.sum(), neg.sum(), m.sum(), (strip & ~m24).sum()))


def auc(s, p, n):
    v = np.concatenate([s[p], s[n]]).astype(np.float64)
    rk = rankdata(v)
    n1, n0 = int(p.sum()), int(n.sum())
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


sig = 192.0 / PX
m32 = m.astype(np.float32)
G = np.maximum(gaussian_filter(m32, sig), 1e-3)
core = m & (gaussian_filter(m32, 2 * sig) > 0.999)
Wv = gaussian_filter(mask.astype(np.float32), 2)


def hp(x):
    return x - gaussian_filter(x * m32, sig) / G


def mblur(x):
    return gaussian_filter(np.where(mask, x, 0).astype(np.float32), 2) / np.maximum(Wv, 1e-3)


kh = {'ScrollPrize': hp(sp), 'Hecate 2.4 rev': hp(h24)}
say('  hp 192 um = %.3f px; core (G(m, 2 sigma) > 0.999) %d px' % (sig, core.sum()))
if not core.any():
    # [machine] the 2-sigma core needs ~124 px to every edge and the strip is 250 px wide, so it is empty; the
    # sanity r is taken over every m px instead (hp itself is edge-normalised by G(m)). Reported, never deciding.
    core = m
    say('  the core is empty on a 250 px strip, so r is taken over all %d m px (no core inset) [machine]' % core.sum())
maps = {}
for n in CKPTS:
    fs = [os.path.join(OUT, 'maps', '%s-c%+d-rev.npy' % (n, c)) for c in (-4, -2, 0, 2, 4)]
    maps[n] = np.mean([np.load(f) for f in fs], 0)
rows = {}
say('')
say('  %-28s %8s | %-14s %-14s' % ('map', 'AUCsup S3', 'hp192 b2 vs SP', 'vs Hecate 2.4'))
for n, a in list(maps.items()) + [('REF ScrollPrize map', sp), ('REF Hecate 2.4 rev', h24)]:
    xh = hp(mblur(a)) if not n.startswith('REF') else None
    r = {'auc_sup_s3': auc(a, pos, neg)}
    if xh is not None:
        for k, v in kh.items():
            r['hp192_b2_vs_' + k] = float(np.corrcoef(xh[core], v[core])[0, 1])
        say('  %-28s %8.4f | %+.4f        %+.4f' % (n, r['auc_sup_s3'], r['hp192_b2_vs_ScrollPrize'], r['hp192_b2_vs_Hecate 2.4 rev']))
    else:
        say('  %-28s %8.4f |' % (n, r['auc_sup_s3']))
    rows[n] = r
r_sh = float(np.corrcoef(kh['ScrollPrize'][core], kh['Hecate 2.4 rev'][core])[0, 1])
say('  beside: hp192 ScrollPrize vs Hecate 2.4, keys unblurred: %+.4f' % r_sh)
json.dump({'crop': [y0, y1, x0, x1], 'strip_px': int(strip.sum()), 'pos': int(pos.sum()), 'neg_sup': int(neg.sum()), 'hp_m_px': int(m.sum()),
           'hp_core_px': int(core.sum()), 'rows': rows, 'sp_vs_hecate_hp192': r_sh,
           'map_sources': {n: {('c%+d' % c): sha256(os.path.join(OUT, 'maps', '%s-c%+d-rev.npy' % (n, c))) for c in (-4, -2, 0, 2, 4)} for n in CKPTS}},
          open(os.path.join(OUT, 'valstrip-score.json'), 'w'), indent=1)
runfile.done(out_txt, '\n'.join(lines) + '\n')
