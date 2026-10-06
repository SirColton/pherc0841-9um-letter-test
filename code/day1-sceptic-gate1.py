"""day1-sceptic-gate1.py - sceptic: robustness of the 252405 model-free gate 1 hp-r half (never computed by the render
part), plus the reference reader's box ink fractions. Model-free only (direct render + ScrollPrize 2.4 um map).
CPU; writes geom/day1-9um/sceptic-numbers/sceptic-gate1.txt only."""
import json
import numpy as np, tifffile
from scipy.ndimage import gaussian_filter

OUT = 'geom/day1-9um/sceptic-numbers/sceptic-gate1.txt'
R = 'geom/day1-9um/render'
PX, E = 9.612, 40
SH = [(dy, dx) for dy in (-312, -208, -104, 0, 104, 208, 312) for dx in (-312, -208, -104, 0, 104, 208, 312) if (dy, dx) != (0, 0)]
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def run(tag, d, k, mask):
    H, W = mask.shape
    M = mask.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
    m = M.astype(np.float32)
    for sig_um in (48.0,):
        sig = sig_um / PX
        g = np.maximum(gaussian_filter(m, sig), 1e-3)
        core = M & (gaussian_filter(m, 2 * sig) > 0.999)
        kh = k - gaussian_filter(k * m, sig) / g
        for blur in (0, 2, 4):
            x = d if not blur else gaussian_filter(np.where(M, d, 0), blur) / np.maximum(gaussian_filter(m, blur), 1e-3)
            xh = x - gaussian_filter(x * m, sig) / g
            r0 = pear(xh[core], kh[core])
            nw, nn = [], []
            for dy, dx in SH:
                kr = np.roll(kh, (dy, dx), (0, 1)); cr = core & np.roll(core, (dy, dx), (0, 1))
                nw.append(pear(xh[cr], kr[cr]))
                ys, ye = max(0, dy), min(H, H + dy); xs, xe = max(0, dx), min(W, W + dx)
                mm = core[ys:ye, xs:xe] & core[ys - dy:ye - dy, xs - dx:xe - dx]
                nn.append(pear(xh[ys:ye, xs:xe][mm], kh[ys - dy:ye - dy, xs - dx:xe - dx][mm]))
            nw, nn = np.array(nw), np.array(nn)
            say('%s hp48 blur %d: r %+.4f | wrap null max %+.4f n>=r %d z %.1f | overlap null max %+.4f n>=r %d z %.1f' % (
                tag, blur, r0, nw.max(), (nw >= r0).sum(), (r0 - nw.mean()) / nw.std(), nn.max(), (nn >= r0).sum(), (r0 - nn.mean()) / nn.std()))
    # quadrant split of the zero-shift hp r (blur 0, 48 um) to see whether one region carries it
    sig = 48.0 / PX
    g = np.maximum(gaussian_filter(m, sig), 1e-3); core = M & (gaussian_filter(m, 2 * sig) > 0.999)
    xh = d - gaussian_filter(d * m, sig) / g; kh = k - gaussian_filter(k * m, sig) / g
    qs = []
    for i0, i1 in ((0, H // 2), (H // 2, H)):
        for j0, j1 in ((0, W // 2), (W // 2, W)):
            c = core[i0:i1, j0:j1]
            qs.append('%+.3f(n%d)' % (pear(xh[i0:i1, j0:j1][c], kh[i0:i1, j0:j1][c]), c.sum()) if c.sum() > 1000 else 'na')
    say('%s hp48 blur 0 by quadrant (TL TR BL BR): %s' % (tag, ' '.join(qs)))


st = np.load(R + '/252405-stack.npy', mmap_mode='r')
run('252405', np.asarray(st[18:23], np.float32).mean(0), np.load(R + '/252405-spmap-grid.npy').astype(np.float32), np.load(R + '/252405-mask.npy'))
st = np.load(R + '/w00big-republished/w00big-stack.npy', mmap_mode='r')
run('w00big', np.asarray(st[18:23], np.float32).mean(0), np.load(R + '/w00big-republished/w00big-spmap-grid.npy').astype(np.float32),
    np.load(R + '/w00big-republished/w00big-mask.npy'))

ink = tifffile.imread(R + '/src/inkmap.tif')
L = json.load(open('geom/day1-9um/reference/letters.json', encoding='utf-8'))
crop = L['render_crop_map_px']
nref = nin = 0
for e in L['letters']:
    r0, r1, c0, c1 = e['box']
    f = float((ink[r0:r1, c0:c1] >= 128).mean())
    cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
    inside = crop['rows'][0] <= cy < crop['rows'][1] and crop['cols'][0] <= cx < crop['cols'][1]
    cnt = e['conf'] in ('clear', 'probable')
    nref += cnt; nin += cnt and inside
    say('%s %-8s centre (%.0f,%.0f) inside %s (json %s) ink>=128 %.3f (json %.3f)' % (
        e['id'], e['conf'], cy, cx, inside, e['centre_in_render_crop'], f, e['ink_frac_in_box_thr128']))
say('N_ref recount %d ; in render crop %d' % (nref, nin))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
