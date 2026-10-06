"""day1-render-mapalign.py - on w00big, where does each 2.4 um map and each label grid sit relative to
ScrollPrize's FULL-RES w00 ink map read by index? Prints only.

WHY. day1-render-labelalign.py found the two 2.4 um maps c0841-9um.py used on w00big (our Hecate 2.4
rev render, and ScrollPrize's map via its ds8 jpg) put the labels' best shift about 8 render px apart.
The full-res ScrollPrize map (23.3 MB tif, fetched by the day-1 gates part into geom/day1-9um/gates/s3,
read here, never written) sits on the w00 bucket canvas at 20 px per cell, so by index it is the
reference for the rest. CPU only; local files.
"""
import numpy as np
import tifffile
import zarr
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

Image.MAX_IMAGE_PIXELS = None
A = 'geom/l0-1203-render/audit/'
H, W, P, E = 995, 1245, 4, 40
R0, C0 = 380 * 20, 500 * 20
mask = np.load(A + 'build-deciding-test/ours-mask.npy')
m24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/mask.png'), np.float32)[:H * P, :W * P].reshape(H, P, W, P).min((1, 3)) > 0
M = mask & m24
M[:E] = M[-E:] = False
M[:, :E] = M[:, -E:] = False
full = tifffile.imread('geom/day1-9um/gates/s3/20260220213127-w00/ink-detection/PHerc0841-20260220213127-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif')
print('full-res w00 map', full.shape, full.dtype, '= %.3f x %.3f the w00 mesh grid (823 x 928)' % (full.shape[0] / 823, full.shape[1] / 928))
spf = full[R0:R0 + H * P, C0:C0 + W * P].astype(np.float32).reshape(H, P, W, P).mean((1, 3))
h24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/hecate-rev.png'), np.float32)[:H * P, :W * P].reshape(H, P, W, P).mean((1, 3))
ink = np.asarray(Image.open('geom/c0841/ink-20260220213127.jpg').convert('L'), np.float32)
gi = 2.5 * (380 + (np.arange(H) * P + (P - 1) / 2) / 20.0) - 0.5
gj = 2.5 * (500 + (np.arange(W) * P + (P - 1) / 2) / 20.0) - 0.5
GJ, GI = np.meshgrid(gj, gi)
spj = map_coordinates(ink, [GI, GJ], order=1)
rep = zarr.open_array('geom/day1-9um/render/labelcheck/w00-20260918/inklabels.zarr/0', mode='r')[:]
rep = rep[R0:R0 + H * P, C0:C0 + W * P].astype(np.float32).reshape(H, P, W, P).mean((1, 3)) / 255
prior = np.load(A + 'verify-nine-um-test-does-it-matter/w00_inklabels_v2-grid.npy')


def r_at(a, b, dy, dx):
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    mm = M[ys:ye, xs:xe] & M[ys - dy:ye - dy, xs - dx:xe - dx]
    return float(np.corrcoef(a[ys:ye, xs:xe][mm], b[ys - dy:ye - dy, xs - dx:xe - dx][mm])[0, 1])


def scan(a, b, R=24):
    a, b = gaussian_filter(a, 2), gaussian_filter(b, 2)
    g = max((r_at(a, b, dy, dx), dy, dx) for dy in range(-R, R + 1, 2) for dx in range(-R, R + 1, 2))
    g = max((r_at(a, b, dy, dx), dy, dx) for dy in range(g[1] - 2, g[1] + 3) for dx in range(g[2] - 2, g[2] + 3))
    return r_at(a, b, 0, 0), g


print('shift (dy, dx) of the SECOND grid that best matches the FIRST, render px of 9.61 um; s2 smoothing; %d px' % M.sum())
for n1, a1, n2, a2 in (('SP full-res by index', spf, 'SP ds8 jpg (c0841 mapping)', spj),
                       ('SP full-res by index', spf, 'Hecate 2.4 rev (ours)', h24),
                       ('SP full-res by index', spf, 'labels republished by index', rep),
                       ('SP full-res by index', spf, 'labels prior grid (labels-on-grid.py)', prior),
                       ('Hecate 2.4 rev (ours)', h24, 'labels republished by index', rep)):
    r0, g = scan(a1, a2)
    print('  %-24s vs %-40s r(0,0) %+.3f  peak %+.3f at (%+d, %+d)' % (n1, n2, r0, g[0], g[1], g[2]))
