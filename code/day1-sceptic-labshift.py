"""day1-sceptic-labshift.py - sceptic: how far is the w00big selection label grid (HF v2 via labels-on-grid.py) from
the 2.4 um ink, and how much does that move the selection AUCs? AUCsup peak over label shifts, for Hecate 2.4 and the
ScrollPrize full-res map; then the 9 um recipes at (0,0) and at each map's peak shift. CPU only (selection frame,
w00big, not held out). Writes geom/day1-9um/sceptic-numbers/sceptic-labshift.txt only."""
import os
import numpy as np
from PIL import Image
from scipy.stats import rankdata

OUT = 'geom/day1-9um/sceptic-numbers/sceptic-labshift.txt'
V = 'geom/l0-1203-render/audit/verify-nine-um-test-does-it-matter'
H, W, P, E = 995, 1245, 4, 40
lab0 = np.load(os.path.join(V, 'w00_inklabels_v2-grid.npy')); sup0 = np.load(os.path.join(V, 'w00_supervision_mask_v2-grid.npy'))
mask = np.load('geom/l0-1203-render/audit/build-deciding-test/ours-mask.npy')
m24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/mask.png'), np.float32)[:H * P, :W * P].reshape(H, P, W, P).min((1, 3)) > 0
M = mask & m24; M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
h24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/hecate-rev.png'), np.float32)[:H * P, :W * P].reshape(H, P, W, P).mean((1, 3))
tif = np.load('geom/day1-9um/render/w00big-republished/w00big-spmap-grid.npy').astype(np.float32)
labR = np.load('geom/day1-9um/render/w00big-republished/w00big-inklabels-grid.npy'); supR = np.load('geom/day1-9um/render/w00big-republished/w00big-supervision-grid.npy')
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def auc(s, pos, neg):
    p = np.sort(s[pos].astype(np.float64)); n = np.sort(s[neg].astype(np.float64))
    lo = np.searchsorted(n, p, 'left'); hi = np.searchsorted(n, p, 'right')
    return float((lo + 0.5 * (hi - lo)).sum() / (len(p) * len(n)))


def sets(l0, s0, dy, dx):
    l = np.roll(l0, (dy, dx), (0, 1)); s = np.roll(s0, (dy, dx), (0, 1))
    return M & (l >= 0.75), M & (l <= 0) & (s >= 0.75)


def scan(x, l0, s0):
    out = {}
    for dy in range(-14, 7, 2):
        for dx in range(-26, 7, 2):
            out[(dy, dx)] = auc(x, *sets(l0, s0, dy, dx))
    k = max(out, key=out.get)
    return out[(0, 0)], k, out[k]


def ld(m, c):
    return np.load('geom/day1-9um/baselines/maps/%s-c%+d-rev.npy' % (m, c)).astype(np.float32)


def rn(a):
    r = np.full(a.shape, 0.5, np.float32); r[M] = (rankdata(a[M]) - 0.5) / M.sum(); return r


m5 = {m: np.mean([ld(m, c) for c in (-4, -2, 0, 2, 4)], 0) for m in ('soup42', 'soup43', 's43_75k')}
R4 = np.mean([rn(m5['soup42']), rn(m5['soup43'])], 0)
peaks = {}
for nm, x in (('Hecate 2.4', h24), ('SP tif', tif)):
    for lname, l0, s0 in (('HF-v2 grid (selection)', lab0, sup0), ('republished by index', labR, supR)):
        z, k, v = scan(x, l0, s0)
        peaks[(nm, lname)] = k
        say('%-10s vs %-24s AUCsup at (0,0) %.4f ; peak %.4f at label shift %s' % (nm, lname, z, v, k))
for lname, l0, s0, ks in (('HF-v2 grid', lab0, sup0, [(0, 0), peaks[('Hecate 2.4', 'HF-v2 grid (selection)')], peaks[('SP tif', 'HF-v2 grid (selection)')]]),
                          ('republished', labR, supR, [(0, 0), peaks[('Hecate 2.4', 'republished by index')], peaks[('SP tif', 'republished by index')]])):
    for k in ks:
        ps, ng = sets(l0, s0, *k)
        say('  %-12s shift %-10s AUCsup: s43_75k %.4f soup42 %.4f soup43 %.4f R4-ondisk %.4f' % (
            lname, k, auc(m5['s43_75k'], ps, ng), auc(m5['soup42'], ps, ng), auc(m5['soup43'], ps, ng), auc(R4, ps, ng)))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
