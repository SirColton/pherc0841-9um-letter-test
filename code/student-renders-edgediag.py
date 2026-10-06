"""student-renders-edgediag.py - where and why the full-w00 render differs from day 1's w00big stack.

student-renders-check.py found the w00big window of the full-w00 render equal to day 1's w00big stack
in its interior (0 differing voxels) and on the centre layer, and different only in the outer 5 px frame
(358,368 voxels, max |diff| 117). HYPOTHESIS, written into student-renders-check.py's docstring before it
ran: day 1 rendered w00big from a 200 x 250 cell CROP (--crop 380 580 500 750), and day1-render-9um.py's
normals() takes np.gradient of the cropped mesh, which is one-sided at the crop's border cells and central
in the full mesh; a render px uses the normals of cells floor(ii), floor(ii) + 1 with ii = (4 i + 1.5) / 20,
so exactly the px with a border cell among those (rows 0-4 and 990-994, cols 0-4 and 1240-1244 of the
window) can differ, and layer 20 (no normal) cannot.
TEST: (1) the normals of the crop and of the full mesh at the same cells, angle between them, border vs
interior cells; (2) the predicted px set vs the px with any differing voxel. Reads w00 files and day 1's
w00big stack only. CPU only. Writes geom/student-9um/renders/w00big-edge-diag-run.txt.
"""
import json

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion

import runfile

OUT = 'geom/student-9um/renders/w00big-edge-diag-run.txt'
runfile.claim(OUT)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


AFF = np.array(json.load(open('reg-pherc0841-affine.json'))['fine_to_coarse_affine'])


def load_mesh(d, crop=None):
    X, Y, Z = (tifffile.imread(d + '/' + k + '.tif').astype(np.float64) for k in 'xyz')
    if crop is not None:
        r0, r1, c0, c1 = crop
        X, Y, Z = X[r0:r1, c0:c1], Y[r0:r1, c0:c1], Z[r0:r1, c0:c1]
    return X, Y, Z, (X > 0) & (Y > 0) & (Z > 0)


def to_coarse(X, Y, Z):
    P = np.stack([Z, Y, X], -1).reshape(-1, 3)
    Q = np.append(P, np.ones((len(P), 1)), 1) @ AFF
    return np.stack([Q[:, k].reshape(X.shape) for k in range(3)], -1)


def normals(Pm, valid):
    N = np.cross(np.gradient(Pm, axis=0), np.gradient(Pm, axis=1))
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    v3 = binary_erosion(valid, np.ones((3, 3), bool), border_value=1)
    N[~v3] = 0
    return N, v3


CROP = (380, 580, 500, 750)
Nc = normals(to_coarse(*load_mesh('geom/c0841/mesh-l0/w00', CROP)[:3]), load_mesh('geom/c0841/mesh-l0/w00', CROP)[3])[0]
Xf, Yf, Zf, vf = load_mesh('geom/c0841/mesh-l0/w00')
Pf = to_coarse(Xf, Yf, Zf)
Pf[~vf] = 0
Nf = normals(Pf, vf)[0][CROP[0]:CROP[1], CROP[2]:CROP[3]]
ang = np.degrees(np.arccos(np.clip((Nc * Nf).sum(-1), -1, 1)))
border = np.zeros(ang.shape, bool)
border[0] = border[-1] = True
border[:, 0] = border[:, -1] = True
say('normals, crop %s of geom/c0841/mesh-l0/w00 vs the full mesh at the same cells (degrees):' % (CROP,))
say('  border cells (%d): median %.3f, max %.3f; cells with angle > 0: %d' % (border.sum(), np.median(ang[border]), ang[border].max(), (ang[border] > 0).sum()))
say('  interior cells (%d): max %.3g; cells with angle > 0: %d' % ((~border).sum(), ang[~border].max(), (ang[~border] > 0).sum()))

h, w = ang.shape
H, W = (h - 1) * 5, (w - 1) * 5
fi = np.floor((np.arange(H) * 4 + 1.5) / 20).astype(int)
fj = np.floor((np.arange(W) * 4 + 1.5) / 20).astype(int)
rows = (fi == 0) | (fi + 1 == h - 1)
cols = (fj == 0) | (fj + 1 == w - 1)
pred = rows[:, None] | cols[None, :]
st = np.load('geom/student-9um/renders/w00-stack.npy', mmap_mode='r')
sub = np.asarray(st[:, 1900:2895, 2500:3745]).astype(np.int16)
ref = np.load('geom/day1-9um/render/w00big-republished/w00big-stack.npy').astype(np.int16)
obs = (sub != ref).any(0)
say('predicted px (a border cell among the two normals bilinear-weighted): %d (rows %s, cols %s)'
    % (pred.sum(), np.nonzero(rows)[0].tolist(), np.nonzero(cols)[0].tolist()))
say('observed px with any differing voxel: %d; observed outside the predicted set: %d; predicted px that happen to round equal on all 41 layers: %d'
    % (obs.sum(), (obs & ~pred).sum(), (pred & ~obs).sum()))
say('layer 20 (the mesh, no normal) differing voxels: %d' % int((sub[20] != ref[20]).sum()))
runfile.done(OUT, '\n'.join(lines) + '\n')
