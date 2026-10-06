"""day1-sceptic-misc.py - sceptic: member hashes vs prereg, soup recompute, w00big self-test, label/map shift scans.
CPU only; reads local files; writes geom/day1-9um/sceptic-numbers/sceptic-misc.txt only."""
import hashlib
import json
import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

OUT = 'geom/day1-9um/sceptic-numbers/sceptic-misc.txt'
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


pre = json.load(open('geom/day1-9um/gates/prereg.json', encoding='utf-8'))
for name, m in pre['members']['ondisk'].items():
    for f, h in zip(m['files'], m['sha256']):
        got = hashlib.sha256(open(f, 'rb').read()).hexdigest()
        say('sha256 %-40s %s' % (f, 'MATCH' if got == h else 'MISMATCH %s' % got[:16]))

for seed in (42, 43):
    ps = [torch.load('geom/ink9um/seed%d-step-%06d.pth' % (seed, s), map_location='cpu', weights_only=False)['model'] for s in (40000, 50000, 60000, 75000)]
    soup = torch.load('geom/day1-9um/baselines/soup%d.pth' % seed, map_location='cpu', weights_only=False)['model']
    dm = d75 = 0.0
    nf = 0
    for k, t in soup.items():
        if t.is_floating_point():
            ref = torch.stack([p[k].double() for p in ps]).mean(0)
            dm = max(dm, float((t.double() - ref).abs().max()))
            d75 = max(d75, float((t.double() - ps[-1][k].double()).abs().max()))
            nf += 1
    say('soup%d: %d float tensors; max |soup - recomputed mean| %.3g; max |soup - 75k| %.4f' % (seed, nf, dm, d75))

a = np.load('geom/day1-9um/render/w00big-republished/w00big-stack.npy', mmap_mode='r')
b = np.load('geom/l0-1203-render/audit/build-deciding-test/ours-stack.npy', mmap_mode='r')
say('w00big self-test: shapes %s %s ; voxels differing %d' % (a.shape, b.shape, int((np.asarray(a) != np.asarray(b)).sum())))


def corr_scan(p, q, M, rng):
    H, W = M.shape
    out = {}
    for dy in rng:
        for dx in rng:
            ys, ye = max(0, dy), min(H, H + dy); xs, xe = max(0, dx), min(W, W + dx)
            mm = M[ys:ye, xs:xe] & M[ys - dy:ye - dy, xs - dx:xe - dx]
            x = p[ys:ye, xs:xe][mm]; y = q[ys - dy:ye - dy, xs - dx:xe - dx][mm]
            out[(dy, dx)] = float(np.corrcoef(x, y)[0, 1])
    k = max(out, key=out.get)
    return out[(0, 0)], k, out[k]


R = 'geom/day1-9um/render'
m5 = np.load(R + '/252405-mask.npy')
lab = gaussian_filter(np.load(R + '/252405-inklabels-grid.npy').astype(np.float32), 4)
sp = gaussian_filter(np.load(R + '/252405-spmap-grid.npy').astype(np.float32), 4)
M5 = m5.copy(); M5[:40] = M5[-40:] = False; M5[:, :40] = M5[:, -40:] = False
r0, k, rk = corr_scan(lab, sp, M5, range(-10, 11, 1))
say('252405 labels(by index) vs SP map, sigma 4, +-10 px: r at (0,0) %.3f ; peak %.3f at %s' % (r0, rk, k))

# w00big: AUC of Hecate 2.4 vs the HF v2 grid (labels-on-grid.py) as the label grid is shifted in columns/rows
V = 'geom/l0-1203-render/audit/verify-nine-um-test-does-it-matter'
H, W, P, E = 995, 1245, 4, 40
lab0 = np.load(os.path.join(V, 'w00_inklabels_v2-grid.npy')); sup0 = np.load(os.path.join(V, 'w00_supervision_mask_v2-grid.npy'))
mask = np.load('geom/l0-1203-render/audit/build-deciding-test/ours-mask.npy')
m24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/mask.png'), np.float32)[:H * P, :W * P].reshape(H, P, W, P).min((1, 3)) > 0
M = mask & m24; M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
h24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/hecate-rev.png'), np.float32)[:H * P, :W * P].reshape(H, P, W, P).mean((1, 3))


def auc(s, pos, neg):
    p = np.sort(s[pos].astype(np.float64)); n = np.sort(s[neg].astype(np.float64))
    lo = np.searchsorted(n, p, 'left'); hi = np.searchsorted(n, p, 'right')
    return float((lo + 0.5 * (hi - lo)).sum() / (len(p) * len(n)))


res = []
for dy in (-4, 0, 4):
    for dx in range(-16, 9, 2):
        l = np.roll(lab0, (dy, dx), (0, 1)); s = np.roll(sup0, (dy, dx), (0, 1))
        res.append((auc(h24, M & (l >= 0.75), M & (l <= 0) & (s >= 0.75)), dy, dx))
res.sort(reverse=True)
say('w00big Hecate 2.4 AUCsup vs HF-v2 grid shifted (label moved by dy,dx): at (0,0) %.4f ; best %s' % (
    [r for r in res if r[1] == 0 and r[2] == 0][0][0], ', '.join('%.4f@(%d,%d)' % r for r in res[:4])))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
