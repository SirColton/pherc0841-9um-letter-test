"""student-renders-hfdiag.py - why the full-w00 HF v2 grids differ from day 1's w00big grids in 4,308 px.

student-renders-labels.py's convention check (the w00big window of the full-w00 HF v2 inklabels grid vs
verify-nine-um-test-does-it-matter/w00_inklabels_v2-grid.npy) found max |diff| 0.25 in 4,308 of 1,238,775
px; the diffs are multiples of 1/16, i.e. raster px of the 4 x 4 gather. HYPOTHESIS, stated before this
ran: the convention gathers raster px at rint(R - 1.5 + a), which sits on half-integers, and np.arange's
start (the dataset window's first row / col, set by the grid's bounding box) moves the float noise that
decides those ties, so a different window shifts the gather block by 1 raster px (2.4 um) without moving
the nearest sub-sample. TEST: rebuild the w00big grid from c0841/mesh0b/w00big with day 1's window (must
equal the saved grid) and with the full-w00 window (27, 785, 3, 937) (must equal the student grid's
window), and count px whose nearest lattice sub-sample, nn distance, rint row or rint col differ.
CPU only; reads w00 files only. Writes geom/student-9um/renders/w00-hf-convention-diag-run.txt.
"""
import runfile
OUT = 'geom/student-9um/renders/w00-hf-convention-diag-run.txt'
runfile.claim(OUT)
lines = []


def print(*a):
    s = ' '.join(str(v) for v in a)
    lines.append(s)
    __builtins__.print(s, flush=True) if hasattr(__builtins__, 'print') else __builtins__['print'](s, flush=True)


import numpy as np, tifffile
from PIL import Image
from scipy.spatial import cKDTree
A='geom/l0-1203-render/audit/'; DS=A+'sota-research/w00-dataset/'
SUB=5; POOL=4
def bil(A_, ii, jj):
    i1 = np.clip(np.floor(ii).astype(int), 0, A_.shape[0] - 2); j1 = np.clip(np.floor(jj).astype(int), 0, A_.shape[1] - 2)
    di, dj = ii - i1, jj - j1
    return (A_[i1, j1] * (1 - di) * (1 - dj) + A_[i1 + 1, j1] * di * (1 - dj) + A_[i1, j1 + 1] * (1 - di) * dj + A_[i1 + 1, j1 + 1] * di * dj)
def grid_points(X, Y, Z):
    h, w = X.shape; H, W = (h - 1) * 20 // POOL, (w - 1) * 20 // POOL
    ii = (np.arange(H) * POOL + (POOL - 1) / 2) / 20.0; jj = (np.arange(W) * POOL + (POOL - 1) / 2) / 20.0
    JJ, II = np.meshgrid(jj, ii); return np.stack([bil(A_, II, JJ) for A_ in (X, Y, Z)], -1)
x, y, z = (np.array(Image.open('geom/c0841/mesh0b/w00big/' + k + '.tif'), np.float64) for k in 'xyz')
Qw = grid_points(x, y, z)
Xd, Yd, Zd = (tifffile.imread(DS + k + '.tif').astype(np.float64) for k in 'xyz')
L = tifffile.imread(DS+'w00_inklabels_v2.tif'); L = L.reshape(L.shape[-2:])
def carry(win):
    r0,r1,c0,c1=win
    RR_, CC_ = np.arange(r0, r1 - 1, 1.0 / SUB), np.arange(c0, c1 - 1, 1.0 / SUB)
    CC, RR = np.meshgrid(CC_, RR_)
    i1, j1 = np.floor(RR).astype(int), np.floor(CC).astype(int)
    ok = (Xd[i1, j1] > 0) & (Xd[i1 + 1, j1] > 0) & (Xd[i1, j1 + 1] > 0) & (Xd[i1 + 1, j1 + 1] > 0)
    P = np.stack([bil(A_, RR, CC)[ok] for A_ in (Xd, Yd, Zd)], 1); dr, dc = RR[ok], CC[ok]
    d, idx = cKDTree(P).query(Qw.reshape(-1, 3), workers=4)
    Rr, Cr = dr[idx].reshape(Qw.shape[:2]) * 20, dc[idx].reshape(Qw.shape[:2]) * 20
    acc = np.zeros(Qw.shape[:2], np.float32)
    for a in range(4):
        for b in range(4):
            acc += L[np.clip(np.rint(Rr - 1.5 + a).astype(int), 0, L.shape[0]-1), np.clip(np.rint(Cr - 1.5 + b).astype(int), 0, L.shape[1]-1)] / 255.0
    return acc/16, dr[idx], dc[idx], d
q = Qw.reshape(-1,3); lo, hi = q.min(0)-60, q.max(0)+60
ok0 = (Xd > 0) & (Xd >= lo[0]) & (Xd <= hi[0]) & (Yd >= lo[1]) & (Yd <= hi[1]) & (Zd >= lo[2]) & (Zd <= hi[2])
rr, cc = np.nonzero(ok0)
wd = (max(rr.min()-2,0), min(rr.max()+3,Xd.shape[0]), max(cc.min()-2,0), min(cc.max()+3,Xd.shape[1]))
old = np.load(A+'verify-nine-um-test-does-it-matter/w00_inklabels_v2-grid.npy')
gA, drA, dcA, dA = carry(wd)
print('day1 window', wd, 'vs saved: max', np.abs(gA-old).max(), 'n', (np.abs(gA-old)>1e-6).sum())
gB, drB, dcB, dB = carry((27, 785, 3, 937))
print('full window (27,785,3,937) vs saved: max', np.abs(gB-old).max(), 'n', (np.abs(gB-old)>1e-6).sum())
lat = (np.rint(drA*5)!=np.rint(drB*5)) | (np.rint(dcA*5)!=np.rint(dcB*5))
print('nearest lattice sub-sample differs at', lat.sum(), 'px; nn dist differs >1e-9 at', (np.abs(dA-dB)>1e-9).sum())
fr = lambda v: v*20-1.5
print('window A Rr-1.5 frac==0.5 exactly:', np.mean(np.abs((fr(drA)%1)-0.5)<1e-12), ' window B:', np.mean(np.abs((fr(drB)%1)-0.5)<1e-12))
print('rint(Rr-1.5) differs at', (np.rint(fr(drA))!=np.rint(fr(drB))).sum(), 'px; rint(Cr-1.5) differs at', (np.rint(fr(dcA))!=np.rint(fr(dcB))).sum())
my = np.load('geom/student-9um/renders/w00-inklabels_v2-grid.npy')[1900:2895,2500:3745]
print('this window-B rebuild vs student grid window: max', np.abs(gB-my).max())
runfile.done(OUT, '\n'.join(lines) + '\n')
