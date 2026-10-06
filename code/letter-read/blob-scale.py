# Correlation length inside each scorable letter box: the radius at which the box's autocorrelation falls to 0.5,
# in render px (9.612 um) and um, for the 2.4 um ceiling map, arm A and the baseline. A map whose half-width grows
# against the ceiling inside the same box has broadened that letter's strokes.
import json, numpy as np, importlib.util, sys
sys.dont_write_bytecode = True
G = r'geom\day1-9um'  # relative to <workdir>
spec = importlib.util.spec_from_file_location('sc', G + r'\letter-read\score.py')
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
refs, _ = sc.load_ref(G + r'\letter-read\252405-ref.json', 9.612)
maps = {'ceiling 2.4': G + r'\render\252405-spmap-grid.npy', 'arm A': G + r'\test\arm-A-R1-tauil_ft6k-rev-mean5.npy',
        'baseline': G + r'\test\baseline-s43_75k-rev-mean5.npy',
        'student': G + r'\..\student-9um\test\student-rev-mean5.npy'}


def half_width(a):
    a = a - a.mean()
    F = np.fft.fft2(a, s=(2 * a.shape[0], 2 * a.shape[1]))
    ac = np.real(np.fft.ifft2(np.abs(F) ** 2))
    ac = np.fft.fftshift(ac) / ac[0, 0]
    cy, cx = np.array(ac.shape) // 2
    y, x = np.indices(ac.shape)
    r = np.hypot(y - cy, x - cx).astype(int)
    prof = np.bincount(r.ravel(), ac.ravel()) / np.bincount(r.ravel())
    below = np.where(prof < 0.5)[0]
    return float(below[0]) if below.size else float('nan')


print('half-width of the autocorrelation inside each letter box, render px (x9.6 um)')
print('%-6s ' % 'ref' + ''.join('%12s' % k for k in maps))
arrs = {k: np.load(v, mmap_mode='r') for k, v in maps.items()}
allw = {k: [] for k in maps}
for r in refs:
    b = [int(round(v)) for v in r['box']]
    row = ''
    for k, a in arrs.items():
        w = half_width(np.asarray(a[max(b[0], 0):b[1] + 1, max(b[2], 0):min(b[3] + 1, a.shape[1])], np.float64))
        allw[k].append(w)
        row += '%12.1f' % w
    print('%-6s ' % r['id'] + row)  # identities are never printed
print('%-7s' % 'median' + ''.join('%12.1f' % np.nanmedian(v) for v in allw.values()))
print('%-7s' % 'med um' + ''.join('%12.0f' % (np.nanmedian(v) * 9.612) for v in allw.values()))
