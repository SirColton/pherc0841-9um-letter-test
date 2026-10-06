"""test_letter_read.py - self-tests for the blind letter read, on SYNTHETIC data only (a fake map with
drawn glyphs, a fake reference, fake reader output). No real map, reference or tile is opened.

Each check prints PASS or FAIL with the outcome that would fail it; the run exits 1 on any FAIL.
  make-pool.py  refusals, opaque names, determinism, real tiles = the map through the card transform,
                phase null keeps the FFT amplitude and the histogram, block null keeps the histogram and
                destroys the glyph, w00big geometry (8 mm tiles, 4 mm overlap, 2 x 2)
  score.py      a correct call matches; a wrong identity does not; a 0.6 mm offset does not while a
                0.45 mm one does; a null call subtracts; overlapping duplicates merge; an upscaled pool
                maps back; a changed reference refuses; unread tiles and foreign tiles refuse
  workflow      test_workflow_mock.mjs (node) runs day1-letter-read-workflow.js on mock agents
Usage: python test_letter_read.py   (CPU only; writes only to a temporary folder)
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.dont_write_bytecode = True      # importing make-pool and score must leave no __pycache__ beside them
for _st in (sys.stdout, sys.stderr):  # a Greek letter in a check message must not crash a cp1252 console
    _st.reconfigure(errors='backslashreplace')
HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, 'make-pool.py')
SCORE = os.path.join(HERE, 'score.py')
PY = sys.executable
PITCH = 9.612
PX_MM = 1000.0 / PITCH
RESULTS = []


def check(cond, what):
    RESULTS.append((bool(cond), what))
    print('%s  %s' % ('PASS' if cond else 'FAIL', what))


def run(script, *args):
    r = subprocess.run([PY, script] + [str(a) for a in args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = load('make_pool', POOL)
sc = load('score', SCORE)
tmp = tempfile.mkdtemp(prefix='letter-read-test-')
try:
    # ---- a fake map: four glyphs drawn on a noisy background, blurred like a model's output -----------
    H, W = 420, 560
    glyphs = [  # (letter, centre i, centre j)
        ('lambda', 150, 160), ('omicron', 120, 330), ('tau', 270, 140), ('pi', 290, 420)]
    img = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(img)
    s = 45
    for name, ci, cj in glyphs:
        if name == 'lambda':
            d.line([(cj - s, ci + s), (cj, ci - s), (cj + s, ci + s)], fill=255, width=10)
        elif name == 'omicron':
            d.ellipse([cj - s, ci - s, cj + s, ci + s], outline=255, width=10)
        elif name == 'tau':
            d.line([(cj - s, ci - s), (cj + s, ci - s)], fill=255, width=10)
            d.line([(cj, ci - s), (cj, ci + s)], fill=255, width=10)
        else:
            d.line([(cj - s, ci + s), (cj - s, ci - s), (cj + s, ci - s), (cj + s, ci + s)], fill=255, width=10)
    ink = np.asarray(img.filter(ImageFilter.GaussianBlur(3)), np.float64) / 255.0
    rng = np.random.default_rng(5)
    fake = (70 + 120 * ink + rng.normal(0, 6, (H, W))).astype(np.float32)
    map_p = os.path.join(tmp, 'fake-map.npy')
    mask_p = os.path.join(tmp, 'fake-mask.npy')
    np.save(map_p, fake)
    np.save(mask_p, np.ones((H, W), bool))
    grid_p = os.path.join(tmp, 'fake-grid.json')
    json.dump(dict(H=H, W=W, pitch_um_in_plane=PITCH), open(grid_p, 'w'))
    common = ['--map', map_p, '--mask', mask_p, '--grid', grid_p, '--tile-mm', '2.5', '--overlap-mm', '1.2',
              '--block-mm', '0.5', '--feather-mm', '0.06']

    # ---- make-pool refusals ----------------------------------------------------------------------------
    t_in = os.path.join(tmp, 'p-in', 'tiles')
    rc, out = run(POOL, *common, '--tiles-dir', t_in, '--key', os.path.join(t_in, 'key.json'), '--seed', 3)
    check(rc != 0 and 'inside the tiles folder' in out and not os.path.exists(t_in),
          'make-pool refuses a key inside the tiles folder and writes nothing (fails if it writes the pool)')

    # ---- the pool ----------------------------------------------------------------------------------------
    tiles1, key1 = os.path.join(tmp, 'p1', 'tiles'), os.path.join(tmp, 'p1-key.json')
    rc, out = run(POOL, *common, '--tiles-dir', tiles1, '--key', key1, '--seed', 3)
    check(rc == 0, 'make-pool runs on the synthetic map (fails on any error)' + ('' if rc == 0 else ': ' + out[-400:]))
    K = json.load(open(key1))
    kt = K['tiles']
    kinds = {}
    for e in kt.values():
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    pngs = sorted(f[:-4] for f in os.listdir(tiles1))
    check(pngs == sorted(kt), 'every key tile has one PNG and the folder holds nothing else')
    # block nulls are off by default since 2026-10-05 (readers named them by their seams)
    check(kinds == {'real': 12, 'phase': 12},
          'default pool: 12 real windows (3 rows x 4 cols) and as many nulls, all phase, no block (got %s)' % kinds)
    import re
    check(all(re.fullmatch(r't[0-9a-f]{7}', n) for n in kt) and not any(k in n for n in kt for k in ('real', 'phase', 'block', 'null')),
          'names are opaque ids t + 7 hex digits with no kind in them')
    order_kinds = [kt[n]['kind'] for n in sorted(kt)]
    check(order_kinds != sorted(order_kinds, key=lambda k: k != 'real') and order_kinds != sorted(order_kinds),
          'alphabetical order of names does not group the kinds (fails if names sort by kind)')
    check(K['meta']['depth_note'] and 'depth' not in kinds, 'no depth null without a depth map, and the key says why')
    check(K['meta'].get('mirror') is True and all(e.get('mirror') is True for e in kt.values() if e['kind'] == 'phase'),
          'phase nulls scramble the mirror extension by default (prereg-v2 note 3; fails if the default is periodic)')
    rc, out = run(POOL, *common, '--tiles-dir', tiles1, '--key', os.path.join(tmp, 'other-key.json'), '--seed', 3)
    check(rc != 0 and 'not empty' in out, 'make-pool refuses a non-empty tiles folder')
    rc, out = run(POOL, *common, '--tiles-dir', os.path.join(tmp, 'p9', 'tiles'), '--key', key1, '--seed', 3)
    check(rc != 0 and 'never overwritten' in out, 'make-pool refuses to overwrite a key')

    # determinism
    tiles2, key2 = os.path.join(tmp, 'p2', 'tiles'), os.path.join(tmp, 'p2-key.json')
    run(POOL, *common, '--tiles-dir', tiles2, '--key', key2, '--seed', 3)
    K2 = json.load(open(key2))
    same_png = all(open(os.path.join(tiles1, n + '.png'), 'rb').read() == open(os.path.join(tiles2, n + '.png'), 'rb').read() for n in kt)
    check(K2['tiles'] == kt and same_png, 'same map, args and seed give the same names, key and PNG bytes')
    tiles3, key3 = os.path.join(tmp, 'p3', 'tiles'), os.path.join(tmp, 'p3-key.json')
    run(POOL, *common, '--tiles-dir', tiles3, '--key', key3, '--seed', 4)
    check(sorted(json.load(open(key3))['tiles']) != sorted(kt), 'another seed gives other names (fails if the seed is ignored)')

    # a 0-1 float map (ScrollPrize's 2.4 um grid is one) is the same picture on the other scale
    unit_p = os.path.join(tmp, 'fake-unit.npy')
    np.save(unit_p, (fake / 255.0).astype(np.float32))
    tilesU, keyU = os.path.join(tmp, 'pU', 'tiles'), os.path.join(tmp, 'pU-key.json')
    rc, out = run(POOL, '--map', unit_p, *common[2:], '--tiles-dir', tilesU, '--key', keyU, '--seed', 3)
    KU = json.load(open(keyU)) if rc == 0 else {'tiles': {}, 'meta': {}}
    diffU = [n for n in kt if kt[n]['kind'] == 'real' and
             np.abs(np.asarray(Image.open(os.path.join(tilesU, n + '.png')), np.int16) -
                    np.asarray(Image.open(os.path.join(tiles1, n + '.png')), np.int16)).max() > 1] if rc == 0 else ['run failed']
    check(rc == 0 and KU['meta'].get('unit_scaled') is True and not diffU,
          'a 0-1 float map gives the 0-255 map\'s real tiles to 1 grey level and the key says unit_scaled '
          '(fails on the all-black pool of 2026-10-05; %s)' % (diffU or 'ok'))
    # a null cut from a window with a masked hole: its valid pixels hold exactly its real twin's valid values,
    # so the scrambled fill never shows inside the valid area (252405's first window has such a corner)
    holed_p = os.path.join(tmp, 'fake-holed-mask.npy')
    holed = np.ones((H, W), bool)
    holed[10:40, 10:40] = False                  # inside the first window, which also gives a null
    np.save(holed_p, holed)
    tilesH, keyH = os.path.join(tmp, 'pH', 'tiles'), os.path.join(tmp, 'pH-key.json')
    rc, out = run(POOL, '--map', map_p, '--mask', holed_p, *common[4:], '--tiles-dir', tilesH, '--key', keyH, '--seed', 3)
    KH = json.load(open(keyH)) if rc == 0 else {'tiles': {}, 'meta': {}}
    pair = {}
    for n, e in KH['tiles'].items():
        if e['masked_frac'] > 0:
            pair[e['kind']] = (n, e)
    same = None
    if set(pair) == {'real', 'phase'}:
        (nr, er), (nn, en) = pair['real'], pair['phase']
        hm = holed[er['origin_i']:er['origin_i'] + er['size_px'][0], er['origin_j']:er['origin_j'] + er['size_px'][1]]
        ur = np.asarray(Image.open(os.path.join(tilesH, nr + '.png')))
        un = np.asarray(Image.open(os.path.join(tilesH, nn + '.png')))
        same = (np.array_equal(np.sort(ur[hm]), np.sort(un[hm])) and not un[~hm].any() and not ur[~hm].any()
                and en['masked_frac'] == round(900.0 / hm.size, 4))
    check(rc == 0 and same is True,
          'a null from a masked window: its valid pixels have the exact valid histogram of its real twin, the hole is black '
          'in both, and the key records masked_frac (fails if the scrambled fill reaches the valid area; got %s)' % same)
    # a map the transform flattens is refused before any tile or key is written
    dark_p = os.path.join(tmp, 'fake-dark.npy')
    np.save(dark_p, (2 + fake / 255.0 * 50).astype(np.float32))
    tiles5, key5 = os.path.join(tmp, 'p5', 'tiles'), os.path.join(tmp, 'p5-key.json')
    rc, out = run(POOL, '--map', dark_p, *common[2:], '--tiles-dir', tiles5, '--key', key5, '--seed', 3)
    check(rc != 0 and 'one flat value' in out and not os.path.exists(key5) and not (os.path.isdir(tiles5) and os.listdir(tiles5)),
          'a map the card transform flattens to one grey is refused with no PNG and no key (exit %d)' % rc)

    # real tiles are the map through the card transform, pixel for pixel
    card = lambda a: np.rint(np.clip((a / 255.0 - 0.25) / 0.5, 0, 1) * 255).astype(np.uint8)
    reals = [n for n in sorted(kt) if kt[n]['kind'] == 'real']
    bad = 0
    for n in reals:
        e = kt[n]
        h, w = e['size_px']
        want = card(fake.astype(np.float64)[e['origin_i']:e['origin_i'] + h, e['origin_j']:e['origin_j'] + w])
        bad += int(not np.array_equal(np.asarray(Image.open(os.path.join(tiles1, n + '.png'))), want))
    check(bad == 0, 'every real tile equals card(map window) at its key origin (%d differ)' % bad)

    # the nulls keep the histogram of their source window, exactly
    hist_bad = 0
    for n in sorted(kt):
        e = kt[n]
        if e['kind'] == 'real':
            continue
        h, w = e['size_px']
        src = card(fake.astype(np.float64)[e['origin_i']:e['origin_i'] + h, e['origin_j']:e['origin_j'] + w])
        got = np.asarray(Image.open(os.path.join(tiles1, n + '.png')))
        hist_bad += int(not np.array_equal(np.sort(got, axis=None), np.sort(src, axis=None)))
        hist_bad += int(np.array_equal(got, src))
    check(hist_bad == 0, 'every null tile has its source window\'s exact pixel histogram and is not the window itself')

    # phase scramble: amplitude kept exactly without the remap; shapes gone
    a = fake.astype(np.float64)[40:300, 40:300]
    q = mp.phase_scramble(a, np.random.default_rng(1), remap=False)
    amp_err = np.max(np.abs(np.abs(np.fft.fft2(q)) - np.abs(np.fft.fft2(a)))) / np.abs(np.fft.fft2(a)).max()
    check(amp_err < 1e-9, 'phase null (no remap) keeps |FFT| to %.1e of its peak (fails above 1e-9)' % amp_err)
    def radial(x, nb=24):
        """radially averaged power spectrum: the blob-size texture a reader sees"""
        P = np.abs(np.fft.fft2(x - x.mean())) ** 2
        r = np.hypot(np.fft.fftfreq(x.shape[0])[:, None], np.fft.fftfreq(x.shape[1])[None, :])
        e = np.linspace(0, 0.5, nb + 1)
        return np.array([P[(r > e[k]) & (r <= e[k + 1])].mean() for k in range(nb)])

    spec_dev = lambda x: float(np.mean(np.abs(np.log10(radial(x) / radial(a)))))
    q2 = mp.phase_scramble(a, np.random.default_rng(1), remap=True)
    r_ph = np.corrcoef(q2.ravel(), a.ravel())[0, 1]
    d_ph, d_1shot = spec_dev(q2), spec_dev(mp.phase_scramble(a, np.random.default_rng(1), True, 0))
    d_shuf = spec_dev(np.random.default_rng(1).permutation(a.ravel()).reshape(a.shape))
    check(np.array_equal(np.sort(q2, axis=None), np.sort(a, axis=None)) and abs(r_ph) < 0.2,
          'phase null (IAAFT) keeps the exact histogram and decorrelates from the glyphs (r %.3f, fails above 0.2)' % r_ph)
    check(d_ph < 0.12 and d_shuf > 0.5 and d_ph < d_1shot,
          'phase null keeps the radial power spectrum: %.3f decades off (fails above 0.12; a pixel shuffle is %.3f, '
          'the one-shot remap %.3f)' % (d_ph, d_shuf, d_1shot))
    check(np.array_equal(mp.phase_scramble(a, np.random.default_rng(1)), q2), 'phase null is deterministic for a seed')

    # --mirror: a tile whose edges do not meet (a ramp) streaks along the axes when scrambled as periodic.
    # Measured as the share of Hann-windowed power above 20 cycles lying on the two axes: 0.7% real, 16% plain
    # null, 1.1% mirror null on this ramp (ink9 tiles: 2.3 / 6.3 / 2.4, 2026-10-05).
    def axis_share(x, kmin=20):
        x = (x - x.mean()) * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
        P = np.abs(np.fft.fft2(x)) ** 2
        KY, KX = np.meshgrid(np.fft.fftfreq(x.shape[0]) * x.shape[0], np.fft.fftfreq(x.shape[1]) * x.shape[1], indexing='ij')
        hi = np.hypot(KY, KX) > kmin
        return P[hi & ((KY == 0) | (KX == 0))].sum() / P[hi].sum()
    yy, xx = np.mgrid[0:200, 0:200]
    rampt = 60 * xx / 200.0 + 40 * yy / 200.0 + np.random.default_rng(1).normal(0, 6, (200, 200))
    s_real = axis_share(rampt)
    s_plain = axis_share(mp.phase_scramble(rampt, np.random.default_rng(9)))
    qm = mp.phase_scramble(rampt, np.random.default_rng(9), mirror=True)
    s_mirror = axis_share(qm)
    check(s_plain > 0.08 and s_mirror < 0.03 and s_real < 0.03,
          'plain phase null streaks on the axes (%.1f%%, fails below 8%%) and the mirror null does not (%.1f%%, fails above 3%%; real %.1f%%)'
          % (100 * s_plain, 100 * s_mirror, 100 * s_real))
    check(np.array_equal(np.sort(qm, axis=None), np.sort(rampt, axis=None)) and qm.shape == rampt.shape,
          'mirror phase null keeps the exact histogram and the tile size')
    d_bl = spec_dev(mp.block_shuffle(a, np.random.default_rng(1), 52, 6, True))
    check(d_bl < 0.12, 'block null keeps the radial power spectrum: %.3f decades off (fails above 0.12)' % d_bl)

    # block shuffle: m = 0 permutes whole blocks; feathered keeps the histogram; the glyph is destroyed
    b = 52
    a2 = fake.astype(np.float64)[0:260, 0:312]
    q0 = mp.block_shuffle(a2, np.random.default_rng(2), b, 0, remap=False)
    blocks = lambda x: sorted(x[i:i + b, j:j + b].tobytes() for i in range(0, 260, b) for j in range(0, 312, b))
    check(blocks(q0) == blocks(a2) and not np.array_equal(q0, a2), 'block null with hard seams is an exact permutation of the blocks')
    w1 = mp.ramp(b, 6)
    check(np.allclose(w1[-12:] + w1[:12], 1.0) and w1[12:-12].min() == 1.0, 'feather ramps of neighbouring blocks sum to 1 across a seam')
    gi, gj = glyphs[0][1], glyphs[0][2]
    g = fake.astype(np.float64)[gi - 104:gi + 104, gj - 104:gj + 104]
    qb = mp.block_shuffle(g, np.random.default_rng(3), b, 6, remap=True)
    r_b = np.corrcoef(qb.ravel(), g.ravel())[0, 1]
    check(np.array_equal(np.sort(qb, axis=None), np.sort(g, axis=None)) and r_b < 0.5,
          'feathered block null keeps the histogram and breaks the glyph (r with the glyph %.3f, fails above 0.5)' % r_b)

    # the w00big geometry the real run uses: 8 mm tiles, 4 mm overlap
    T, O = int(round(8000 / PITCH)), int(np.ceil(4000 / PITCH))
    rows, th = mp.positions(995, T, O)
    cols, tw = mp.positions(1245, T, O)
    check((rows, cols, th, tw) == ([0, 163], [0, 413], 832, 832), 'w00big: 8 mm tiles land at rows %s cols %s (want [0, 163] [0, 413])' % (rows, cols))
    check(min(th - (rows[1] - rows[0]), tw - (cols[1] - cols[0])) >= O, 'w00big: neighbours overlap by >= 4 mm, so a 4 mm letter is whole in some tile')

    # ---- score.py ----------------------------------------------------------------------------------------
    ref_p = os.path.join(tmp, 'ref.json')
    ref = dict(letters=[dict(id='g%d' % k, letter=L, i=ci, j=cj, x_mm=cj * PITCH / 1000, y_mm=ci * PITCH / 1000,
                             box_px=[ci - 70, ci + 70, cj - 70, cj + 70], confidence='high')
                        for k, (L, ci, cj) in enumerate(glyphs)])
    json.dump(ref, open(ref_p, 'w'), indent=1)
    rc, out = run(SCORE, 'lock', '--reference', ref_p)
    check(rc == 0 and os.path.exists(ref_p + '.sha256'), 'lock writes the sha256 file')
    rc, out = run(SCORE, 'lock', '--reference', ref_p)
    check(rc == 4 and 'written once' in out, 'a second lock is refused')

    # --windows-from: a development subset, the windows that hold a reference letter whole, chosen by geometry
    tilesW, keyW = os.path.join(tmp, 'pW', 'tiles'), os.path.join(tmp, 'pW-key.json')
    rc, out = run(POOL, *common, '--tiles-dir', tilesW, '--key', keyW, '--seed', 3, '--windows-from', ref_p)
    KW = json.load(open(keyW)) if rc == 0 else {'tiles': {}, 'meta': {}}
    th_, tw_ = KW['meta'].get('tile_px', [0, 0])
    allpos = [(i, j) for i in KW['meta'].get('row_origins', []) for j in KW['meta'].get('col_origins', [])]
    keptpos = {(e['origin_i'], e['origin_j']) for e in KW['tiles'].values() if e['kind'] == 'real'}
    letterpos = {(e['origin_i'], e['origin_j']) for e in KW['tiles'].values() if e['kind'] == 'real' and e.get('subset') == 'letter'}
    emptypos = {(e['origin_i'], e['origin_j']) for e in KW['tiles'].values() if e['kind'] == 'real' and e.get('subset') == 'empty'}
    bxs = [L['box_px'] for L in ref['letters']]
    holdsW = lambda p: any(p[0] <= b[0] and b[1] < p[0] + th_ and p[1] <= b[2] and b[3] < p[1] + tw_ for b in bxs)
    touchW = lambda p: any(p[0] <= b[1] and b[0] < p[0] + th_ and p[1] <= b[3] and b[2] < p[1] + tw_ for b in bxs)
    eligibleW = [p for p in allpos if not holdsW(p) and not touchW(p)]
    check(rc == 0 and letterpos and letterpos == {p for p in allpos if holdsW(p)} and len(letterpos) < len(allpos)
          and keptpos == letterpos | emptypos and not (letterpos & emptypos)
          and all((e['origin_i'], e['origin_j']) in keptpos for e in KW['tiles'].values()),
          '--windows-from keeps exactly the windows holding a letter whole (%d of %d), nulls only from kept windows '
          '(fails if a window is kept or dropped wrongly)' % (len(letterpos), len(allpos)))
    check(not any(touchW(p) for p in emptypos) and len(emptypos) == min(len(letterpos), len(eligibleW))
          and KW['meta'].get('empty_windows') == len(emptypos),
          'its EMPTY real windows touch no reference box and number min(letter windows, eligible) = %d '
          '(fails if an empty window holds part of a letter)' % len(emptypos))
    # the same with a one-letter reference, so empty windows exist and the rule is exercised, not vacuous
    ref1_p = os.path.join(tmp, 'ref-one.json')
    json.dump(dict(letters=ref['letters'][:1]), open(ref1_p, 'w', encoding='utf-8'))
    tilesW1, keyW1 = os.path.join(tmp, 'pW1', 'tiles'), os.path.join(tmp, 'pW1-key.json')
    rc, out = run(POOL, *common, '--tiles-dir', tilesW1, '--key', keyW1, '--seed', 3, '--windows-from', ref1_p)
    KW1 = json.load(open(keyW1)) if rc == 0 else {'tiles': {}, 'meta': {}}
    b1 = ref['letters'][0]['box_px']
    lp1 = {(e['origin_i'], e['origin_j']) for e in KW1['tiles'].values() if e['kind'] == 'real' and e.get('subset') == 'letter'}
    ep1 = {(e['origin_i'], e['origin_j']) for e in KW1['tiles'].values() if e['kind'] == 'real' and e.get('subset') == 'empty'}
    touch1 = lambda p: p[0] <= b1[1] and b1[0] < p[0] + th_ and p[1] <= b1[3] and b1[2] < p[1] + tw_
    check(rc == 0 and lp1 and len(ep1) == len(lp1) and not any(touch1(p) for p in ep1) and not (lp1 & ep1),
          'with one letter: %d letter windows and as many empty ones, none touching its box (fails if empties are '
          'missing, too few, or overlap the letter)' % len(lp1))

    def holder(i, j, exclude=()):
        """a real tile that holds (i, j) with 60 px to spare, and the position in it"""
        for n in reals:
            e = kt[n]
            h, w = e['size_px']
            ti, tj = i - e['origin_i'], j - e['origin_j']
            if n not in exclude and 60 <= ti <= h - 60 and 60 <= tj <= w - 60:
                return n, ti, tj
        raise AssertionError('no tile holds %s' % ((i, j),))

    def call(tile, letter, ti, tj, half=40, conf=0.8):
        return dict(tile=tile, letter=letter, box=[tj - half, ti - half, tj + half, ti + half], confidence=conf)

    nulls = [n for n in sorted(kt) if kt[n]['kind'] != 'real']
    off6, off45 = 0.6 * PX_MM, 0.45 * PX_MM
    (L0, i0, j0), (L1, i1, j1), (L2, i2, j2), (L3, i3, j3) = glyphs
    t0, a0, b0 = holder(i0, j0)
    t0b, a0b, b0b = holder(i0, j0, exclude=(t0,))
    t1, a1, b1 = holder(i1, j1)
    t2, a2_, b2 = holder(i2 + off6, j2)
    t3, a3, b3 = holder(i3, j3 + off45)
    calls = [
        call(t0, 'lambda', a0, b0),                 # correct: matches
        call(t0b, 'Λ', a0b + 3, b0b - 2, conf=0.5),  # the same letter from an overlapping tile: merges
        call(t1, 'theta', a1, b1),                  # wrong identity on omicron's centre: no match
        call(t2, 'tau', a2_, b2),                   # 0.6 mm below tau: no match
        call(t3, 'pi', a3, b3),                     # 0.45 mm right of pi: matches
        call(nulls[0], 'alpha', 100, 100),          # a call on a null tile: subtracts
    ]
    reads_p = os.path.join(tmp, 'reads.json')
    json.dump(dict(tiles_read=sorted(kt), calls=calls), open(reads_p, 'w'))
    rep_p = os.path.join(tmp, 'report.json')
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p, '--out', rep_p)
    check(rc == 0, 'score runs (fails on any refusal)' + ('' if rc == 0 else ': ' + out[-400:]))
    R = json.load(open(rep_p))
    got = {m['ref_id']: m['dist_mm'] for m in R['matches']}
    check('g0' in got and got['g0'] < 0.05, 'a correct call matches its reference letter (g0 at %.3f mm)' % got.get('g0', -1))
    check('g1' not in got, 'a wrong identity on the exact centre does not match')
    check('g2' not in got, 'a correct identity 0.6 mm off does not match')
    check('g3' in got and abs(got['g3'] - 0.45) < 0.02, 'a correct identity 0.45 mm off matches (%.3f mm)' % got.get('g3', -1))
    check(R['matched'] == 2 and R['real_calls_raw'] == 5 and R['real_calls'] == 4,
          'the overlapping duplicate merges: 5 raw real calls -> 4, matched 2 (got %d -> %d, %d)' % (R['real_calls_raw'], R['real_calls'], R['matched']))
    check(abs(R['precision'] - 0.5) < 1e-9, 'precision = matched / merged real calls = 2/4 (got %s)' % R['precision'])
    nk = kt[nulls[0]]['kind']
    check(R['null_calls'] == 1 and R['null_calls_by_kind'].get(nk) == 1, 'the null call is counted under its kind (%s)' % nk)
    check(R['net_letters'] == 1, 'NET LETTERS = 2 matched - 1 null call = 1 (got %s)' % R['net_letters'])
    # LOCATED, the diagnostic: identity ignored, so the misnamed g1 counts and the 0.6 mm g2 does not
    check('g1' in R['located_ids'] and 'g2' not in R['located_ids'] and {'g0', 'g3'} <= set(R['located_ids']),
          'LOCATED counts a misnamed letter on its centre and not one 0.6 mm off (got %s)' % R['located_ids'])
    # LOCATED-BOX, the second diagnostic: centre inside the letter's box (here +-70 px, 0.68 mm, so 0.5-0.68 mm is box-only), any identity
    check(R['located_box_ids'] == ['g0', 'g1', 'g2', 'g3'] and R['located_box_of'] == 4,
          'LOCATED-BOX counts the 0.6 mm g2 because it sits inside the tau box (got %s of %s)' % (R['located_box_ids'], R['located_box_of']))
    tin, ain, bin_ = holder(i1, j1 + 65)       # misnamed, 0.62 mm right of omicron's centre, inside its box
    tout, aout, bout = holder(i0, j0 + 72)     # 0.69 mm right of lambda's centre, past its box edge (70.5)
    json.dump(dict(tiles_read=sorted(kt), calls=[call(tin, 'sigma', ain, bin_), call(tout, 'lambda', aout, bout)]), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p, '--out', rep_p)
    R7 = json.load(open(rep_p))
    check(rc == 0 and R7['located_box_ids'] == ['g1'] and R7['located_ids'] == [] and R7['matched'] == 0,
          'LOCATED-BOX takes a misnamed call inside the box and not one 1.5 px past its edge; LOCATED takes neither '
          '(got box %s, radius %s)' % (R7.get('located_box_ids'), R7.get('located_ids')))
    ref_o = os.path.join(tmp, 'ref-order.json')
    bad_order = json.loads(json.dumps(ref))
    g = bad_order['letters'][0]
    g['box_px'] = [g['i'] - 90, g['j'] - 90, g['i'] + 90, g['j'] + 90]   # [i0, j0, i1, j1]: not this file's order
    json.dump(bad_order, open(ref_o, 'w'), indent=1)
    run(SCORE, 'lock', '--reference', ref_o)
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_o, '--out', rep_p)
    R8 = json.load(open(rep_p))
    check(rc == 0 and R8['located_box_of'] == 3 and R8['located_box_ids'] == ['g1'] and 'does not hold its centre' in out,
          'a reference box in another order is refused for LOCATED-BOX, said aloud, never read as a box (got %s of %s)'
          % (R8.get('located_box_ids'), R8.get('located_box_of')))

    # without the merge the duplicate would be a false call: the rule bites (fails if merge is a no-op)
    far = [dict(c) for c in calls]
    di = 78 if a0b + 78 <= kt[t0b]['size_px'][0] - 60 else -78       # 0.75 mm from the first call, inside its tile
    far[1] = call(t0b, 'Λ', a0b + di, b0b, conf=0.5)
    json.dump(dict(tiles_read=sorted(kt), calls=far), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p, '--out', rep_p)
    R2 = json.load(open(rep_p))
    check(R2['real_calls'] == 5 and R2['matched'] == 2 and not R2['invalid'],
          'a same-letter call 0.75 mm away does not merge (5 calls, matched 2; got %d, %d)' % (R2['real_calls'], R2['matched']))

    # one reference letter, two same-letter calls 0.4 mm either side of it (0.8 mm apart, so they do not
    # merge): one-to-one matches one of them, the other is a false call
    off4 = 0.4 * PX_MM
    ta, aa, ba = holder(i1, j1 - off4)
    tb, ab, bb = holder(i1, j1 + off4)
    json.dump(dict(tiles_read=sorted(kt), calls=[call(ta, 'omicron', aa, ba), call(tb, 'omicron', ab, bb)]), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p, '--out', rep_p)
    R6 = json.load(open(rep_p))
    check(R6['real_calls'] == 2 and R6['matched'] == 1, 'a reference letter is matched at most once (2 calls 0.4 mm either side: matched %d of %d)' % (R6['matched'], R6['real_calls']))

    # a second null call on another kind, and a duplicate on the same null tile merging within it
    more = calls + [call(nulls[0], 'alpha', 102, 101), call(nulls[-1], 'omega', 50, 60)]
    json.dump(dict(tiles_read=sorted(kt), calls=more), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p, '--out', rep_p)
    R3 = json.load(open(rep_p))
    check(R3['null_calls'] == 2 and R3['net_letters'] == 0, 'null calls: a same-spot duplicate merges within its tile, another tile adds one (2, NET 0; got %s, %s)' % (R3['null_calls'], R3['net_letters']))

    # an upscaled pool maps png px back through the scale
    tiles4, key4 = os.path.join(tmp, 'p4', 'tiles'), os.path.join(tmp, 'p4-key.json')
    run(POOL, *common, '--tiles-dir', tiles4, '--key', key4, '--seed', 3, '--upscale', 2)
    k4 = json.load(open(key4))['tiles']
    n4 = [n for n in sorted(k4) if k4[n]['kind'] == 'real' and
          60 <= i0 - k4[n]['origin_i'] <= k4[n]['size_px'][0] - 60 and 60 <= j0 - k4[n]['origin_j'] <= k4[n]['size_px'][1] - 60][0]
    pi_, pj_ = (i0 - k4[n4]['origin_i'] + 0.5) * 2 - 0.5, (j0 - k4[n4]['origin_j'] + 0.5) * 2 - 0.5
    with Image.open(os.path.join(tiles4, n4 + '.png')) as im:
        size_ok = [im.height, im.width] == k4[n4]['png_px'] == [2 * k4[n4]['size_px'][0], 2 * k4[n4]['size_px'][1]]
    json.dump(dict(tiles_read=sorted(k4), calls=[call(n4, 'lambda', pi_, pj_, half=80)]), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key4, '--reference', ref_p, '--out', rep_p)
    R4 = json.load(open(rep_p))
    check(size_ok and R4['matched'] == 1 and R4['matches'][0]['dist_mm'] < 0.01, 'a 2x upscaled tile maps back to the same w00big point')

    # refusals
    json.dump(dict(tiles_read=sorted(kt), calls=calls), open(reads_p, 'w'))
    changed = json.load(open(ref_p))
    changed['letters'][1]['letter'] = 'theta'
    ref_bak = open(ref_p, 'rb').read()
    json.dump(changed, open(ref_p, 'w'), indent=1)
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p)
    check(rc == 4 and 'REFUSED' in out and 'not the locked one' in out, 'a changed reference is refused (exit %d)' % rc)
    open(ref_p, 'wb').write(ref_bak)
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p)
    check(rc == 0, 'the restored reference scores again')
    ref2 = os.path.join(tmp, 'ref-unlocked.json')
    shutil.copy(ref_p, ref2)
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref2)
    check(rc == 4 and 'no lock file' in out, 'a reference with no lock is refused')
    # the reference maker's own lock form: <stem>.lock.json holding {"sha256": ...}
    ref_j = os.path.join(tmp, 'ref-j.json')
    shutil.copy(ref_p, ref_j)
    json.dump(dict(file='ref-j.json', sha256=sc.sha256(ref_j), letters=4), open(os.path.join(tmp, 'ref-j.lock.json'), 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_j)
    rcl, outl = run(SCORE, 'lock', '--reference', ref_j)
    json.dump(changed, open(ref_j, 'w'), indent=1)
    rc2, out2 = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_j)
    check(rc == 0 and rcl == 4 and rc2 == 4 and 'not the locked one' in out2,
          'a JSON .lock.json is found and honoured: scores (exit %d), blocks a second lock (exit %d), refuses a changed reference (exit %d)' % (rc, rcl, rc2))
    json.dump(dict(tiles_read=sorted(kt)[1:], calls=calls), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p)
    rc2, out2 = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p, '--allow-unread')
    check(rc == 6 and rc2 == 0, 'an unread tile refuses (exit %d), and --allow-unread scores anyway (exit %d)' % (rc, rc2))
    json.dump(dict(tiles_read=sorted(kt), calls=calls + [call('tffffff0', 'alpha', 10, 10)]), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p)
    check(rc == 3, 'a call on a tile that is not in the key refuses (exit %d)' % rc)
    json.dump(dict(tiles_read=sorted(kt), calls=[call(t0, 'lambda', 1000, 1000)]), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_p, '--out', rep_p)
    R5 = json.load(open(rep_p))
    check(R5['matched'] == 0 and R5['real_calls'] == 1 and R5['invalid'], 'a box off its own tile counts as a call that cannot match')

    # manifest and count
    rc, out = run(SCORE, 'manifest', '--key', key1, '--tiles-dir', tiles1, '--reference', ref_p)
    M = json.loads(out.strip().splitlines()[-1])
    check(rc == 0 and M['ok'] and sum(t['control'] for t in M['tiles']) == 12 and len(M['tiles']) == 24,
          'manifest: ok, 24 tiles, 12 controls')
    rc, out = run(SCORE, 'manifest', '--key', key1, '--tiles-dir', tiles1, '--reference', ref2)
    check(rc == 5 and 'REFERENCE NOT LOCKED' in out, 'manifest refuses an unlocked reference (exit %d)' % rc)
    open(os.path.join(tiles1, 'stray.txt'), 'w').write('x')
    rc, out = run(SCORE, 'manifest', '--key', key1, '--tiles-dir', tiles1, '--reference', ref_p)
    os.remove(os.path.join(tiles1, 'stray.txt'))
    check(rc == 5 and 'non-PNG' in out, 'manifest refuses a tiles folder holding anything else')
    json.dump(dict(reads=[{}, {}], calls=calls), open(reads_p, 'w'))
    rc, out = run(SCORE, 'count', '--reads', reads_p)
    C = json.loads(out)
    check(C == dict(n_calls=len(calls), box_sum=int(sum(round(v) for c in calls for v in c['box'])), n_reads=2), 'count prints calls, box sum and reads')

    # letter names
    cases = {'α': 'alpha', 'Λ': 'lambda', 'C': 'sigma', 'ϲ': 'sigma', 'lamda': 'lambda', 'Sigma (lunate)': 'sigma',
             'OMICRON': 'omicron', 'ή': 'eta', '?': None, 'unclear': None}
    wrong = {k: (sc.norm_letter(k), v) for k, v in cases.items() if sc.norm_letter(k) != v}
    check(not wrong, 'letter names normalise (Greek characters, names, homoglyphs, accents; unknown -> None) %s' % (wrong or ''))

    # ---- a reference letter's listed alternatives match (prereg v2: "Reader A's listed alternatives accepted")
    ref_a = os.path.join(tmp, 'ref-alt.json')
    alt_ref = json.loads(json.dumps(ref))
    alt_ref['letters'][1]['alt'] = ['Θ']                 # omicron, or theta
    alt_ref['letters'][0]['alt'] = ['Λ + Δ']        # a two-letter reading: never an alternative
    json.dump(alt_ref, open(ref_a, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    run(SCORE, 'lock', '--reference', ref_a)
    json.dump(dict(tiles_read=sorted(kt), calls=calls), open(reads_p, 'w'))
    rc, out = run(SCORE, 'score', '--reads', reads_p, '--key', key1, '--reference', ref_a, '--out', rep_p)
    RA = json.load(open(rep_p)) if rc == 0 else {'matches': []}
    via = {m['ref_id']: m.get('via_alt') for m in RA['matches']}
    check(rc == 0 and via.get('g1') is True and RA['matched'] == 3 and RA['net_letters'] == 2,
          'a call naming a listed alternative (theta for omicron) matches, marked via_alt (3 matched, NET 2; got %s, %s)'
          % (RA.get('matched'), via))
    check('names more than one letter' in out and via.get('g0') is False,
          'a two-letter alternative ("Lambda + Delta") is refused aloud and g0 still matches as itself')

    # ---- note4-verdict.py's GO rule (prereg-v2 note 4 and its clarification) ------------------------------
    n4 = load('note4', os.path.join(HERE, 'note4-verdict.py'))
    rep4 = lambda m, ph, dp, prec, tiles=None: dict(matched=m, precision=prec, null_calls_by_kind=dict(phase=ph, depth=dp),
                                                    tiles_read_by_kind=tiles or dict(real=25, phase=25, depth=25))
    check(n4.judge(rep4(6, 24, 24, 0.11))['go'] is True, 'note 4: 6 matched, 24 and 24 null calls, precision 0.11 is GO (fails if a bound is strict)')
    check(n4.judge(rep4(5, 0, 0, 1.0))['go'] is False, 'note 4: 5 matched is NO GO however clean')
    check(n4.judge(rep4(8, 25, 0, 0.9))['go'] is False and n4.judge(rep4(8, 0, 25, 0.9))['go'] is False,
          'note 4: 25 calls on EITHER null type alone is NO GO (each type capped separately)')
    check(n4.judge(rep4(8, 0, 0, 0.10))['go'] is False, 'note 4: precision 0.10 is NO GO')
    check(n4.judge(rep4(8, 12, 0, 0.9, dict(real=25, phase=12, depth=25)))['go'] is False,
          'note 4: 12 calls on 12 phase tiles scale to 25 per 25 tiles, NO GO (fails if counts are not scaled)')

    # ---- ref-from-letters.py: a Reader A letters.json (the 252405 form, synthetic here) ---------------------
    RFL = os.path.join(HERE, 'ref-from-letters.py')
    lj = dict(box_order='[row0, row1, col0, col1] in map pixels of x.tif, half-open, row 0 at the top', map_pixel_um=2.403,
              render_crop_map_px={'rows': [840, 1240], 'cols': [4200, 4680], 'from': 'grid.json, pool 4, H 100, W 120'},
              summary=dict(N_ref_centre_in_render_crop=1),
              letters=[dict(id='A', char='Κ', conf='clear', counted_in_N_ref=True, centre_in_render_crop=True,
                            box=[880, 920, 4320, 4400]),
                       dict(id='B', char='Ω', conf='possible', counted_in_N_ref=False, centre_in_render_crop=True,
                            box=[848, 864, 4208, 4224]),
                       dict(id='C', char='Τ', conf='probable', counted_in_N_ref=True, centre_in_render_crop=False,
                            box=[1140, 1180, 4800, 4840])])
    lj_p, lref_p = os.path.join(tmp, 'letters.json'), os.path.join(tmp, 'ref-252405.json')
    json.dump(lj, open(lj_p, 'w', encoding='utf-8'), ensure_ascii=False)
    rc, out = run(RFL, '--letters', lj_p, '--out', lref_p)
    LR = json.load(open(lref_p, encoding='utf-8')) if rc == 0 else {'letters': [], 'not_scored': []}
    a = LR['letters'][0] if LR['letters'] else {}
    # map rows [880, 920) are render edges 10..20, centre edge 15 = render pixel centre 14.5; cols 4320..4400 -> 30..50
    check(rc == 0 and len(LR['letters']) == 1 and (a.get('i'), a.get('j'), a.get('box_px')) == (14.5, 39.5, [10.0, 19.0, 30.0, 49.0]),
          'ref-from-letters: map box -> render centre (14.5, 39.5) and inclusive box [10, 19, 30, 49] (got %s %s %s)'
          % (a.get('i'), a.get('j'), a.get('box_px')))
    check(sorted(x['id'] for x in LR['not_scored']) == ['B', 'C'],
          'ref-from-letters: a possible letter and one centred outside the crop are not scored (got %s)' % [x['id'] for x in LR['not_scored']])
    lrefs, lnotes = sc.load_ref(lref_p, LR.get('pitch_um', 9.612))
    edges = ((a['box_px'][0] - 0.5 + 0.5) * 4 + 840, (a['box_px'][1] + 0.5 + 0.5) * 4 + 840) if a else None
    check(len(lrefs) == 1 and lrefs[0]['letter'] == 'kappa' and lrefs[0]['box'] is not None and not lnotes and edges == (880.0, 920.0),
          'score.py reads it: kappa, box kept, and its half-pixel edges land on the half-open map box (got %s, %s)' % (edges, lnotes))
    rc, out = run(RFL, '--letters', lj_p, '--out', lref_p)
    check(rc != 0 and 'written once' in out, 'ref-from-letters refuses to overwrite a reference')
    bad = json.loads(json.dumps(lj))
    bad['letters'][0]['centre_in_render_crop'] = False
    bad_p = os.path.join(tmp, 'letters-bad.json')
    json.dump(bad, open(bad_p, 'w', encoding='utf-8'), ensure_ascii=False)
    rc, out = run(RFL, '--letters', bad_p, '--out', os.path.join(tmp, 'ref-bad.json'))
    check(rc != 0 and 'letters.json says' in out and not os.path.exists(os.path.join(tmp, 'ref-bad.json')),
          'ref-from-letters refuses when its own crop count disagrees with letters.json, writing nothing')
    bad['letters'][0]['centre_in_render_crop'] = True
    bad['box_order'] = '[row0, row1, col0, col1] in map pixels, inclusive'
    json.dump(bad, open(bad_p, 'w', encoding='utf-8'), ensure_ascii=False)
    rc, out = run(RFL, '--letters', bad_p, '--out', os.path.join(tmp, 'ref-bad.json'))
    check(rc != 0 and 'box_order' in out, 'ref-from-letters refuses a box order it was not written for (inclusive, not half-open)')

    # ---- w00ref-merge.py: nine region readers' marks -> a letters.json ref-from-letters.py accepts -----------
    MERGE = os.path.join(HERE, 'w00ref-merge.py')
    mk = lambda k, core, letters: dict(region=dict(k=k, core=core, window=core), marks=dict(
        orientation=dict(mirrored=False, upside_down=False, evidence='x'), letters=letters, notes=''))
    cores = [[a * 1370, (a + 1) * 1370, b * 1545, (b + 1) * 1545] for a in range(3) for b in range(3)]
    good = [mk(k, c, []) for k, c in enumerate(cores)]
    good[4]['marks']['letters'] = [dict(line=0, char='Κ', alt=[], conf='clear', box=[2000, 2300, 2000, 2280], note=''),
                                   dict(line=0, char='Ω', alt=['Υ'], conf='possible', box=[2000, 2200, 2400, 2600], note='')]
    mpath, lpath, rpath = (os.path.join(tmp, x) for x in ('w00-marks.json', 'w00-letters.json', 'w00-ref.json'))
    json.dump(good, open(mpath, 'w', encoding='utf-8'), ensure_ascii=False)
    rc, out = run(MERGE, '--marks', mpath, '--out', lpath)
    rc2, out2 = run(RFL, '--letters', lpath, '--out', rpath) if rc == 0 else (1, '')
    WR = json.load(open(rpath, encoding='utf-8')) if rc2 == 0 else {'letters': [], 'not_scored': []}
    k0 = WR['letters'][0] if WR['letters'] else {}
    check(rc == 0 and rc2 == 0 and len(WR['letters']) == 1 and (k0.get('i'), k0.get('j'), k0.get('box_px')) == (2149.5, 2139.5, [2000.0, 2299.0, 2000.0, 2279.0])
          and [x['id'] for x in WR['not_scored']] == ['R4-2'],
          'w00ref-merge then ref-from-letters: a half-open render box [2000,2300)x[2000,2280) centres at (2149.5, 2139.5), '
          'the possible mark is not scored (got %s %s %s; %s)' % (k0.get('i'), k0.get('j'), k0.get('box_px'), (out + out2)[-200:]))
    bad = json.loads(json.dumps(good))
    bad[0]['marks']['letters'] = [dict(line=0, char='Α', alt=[], conf='clear', box=[1300, 1500, 100, 300], note='')]
    json.dump(bad, open(mpath, 'w', encoding='utf-8'), ensure_ascii=False)
    lbad = os.path.join(tmp, 'w00-letters-bad.json')
    rc, out = run(MERGE, '--marks', mpath, '--out', lbad)
    check(rc != 0 and 'outside region 0 core' in out and not os.path.exists(lbad),
          'w00ref-merge refuses a mark centred outside its own reader\'s core (a duplicate in the making), writing nothing')

    # ---- w00ref-merge2.py: two readers per region, kept by agreement, deduped across regions -----------------
    MERGE2 = os.path.join(HERE, 'w00ref-merge2.py')
    m2 = load('merge2', MERGE2)
    regs = []
    for a_ in range(3):
        for b_ in range(3):
            core = [a_ * 5480, (a_ + 1) * 5480, b_ * 6180, (b_ + 1) * 6180]
            regs.append(dict(k=a_ * 3 + b_, core=core, zone=[max(0, core[0] - 400), min(16440, core[1] + 400), max(0, core[2] - 400), min(18540, core[3] + 400)]))
    Lt = lambda ch, conf, box, alt=(): dict(char=ch, alt=list(alt), conf=conf, box=box, note='')
    runs2 = [dict(region=r, reader=x, marks=dict(letters=[], notes='')) for r in regs for x in 'ab']
    get = lambda k, x: [r for r in runs2 if r['region']['k'] == k and r['reader'] == x][0]['marks']['letters']
    get(4, 'a').extend([Lt('Κ', 'clear', [8000, 9000, 8000, 8800]), Lt('Ω', 'possible', [6000, 6800, 9000, 9800], ['Υ']),
                        Lt('Α', 'probable', [7000, 7500, 10000, 10500]), Lt('Τ', 'probable', [7000, 8000, 12200, 12600])])
    get(4, 'b').extend([Lt('Κ', 'possible', [8050, 9050, 8020, 8820]), Lt('Υ', 'probable', [6020, 6820, 9010, 9810]),
                        Lt('Λ', 'probable', [7000, 7500, 10000, 10500]), Lt('Τ', 'clear', [7010, 8010, 12190, 12610])])
    get(5, 'a').append(Lt('Τ', 'probable', [7000, 8000, 12210, 12590]))
    get(5, 'b').append(Lt('Τ', 'possible', [7020, 7990, 12200, 12600]))
    kept2, dropped2, unpaired2, deduped2 = m2.merge(runs2)
    chars2 = sorted(L['char'] for L in kept2)
    kap = [L for L in kept2 if L['char'] == 'Κ']
    tau = [L for L in kept2 if L['char'] == 'Τ']
    check(chars2 == sorted(['Κ', 'Υ', 'Τ']) and len(unpaired2) == 2 and len(deduped2) == 1,
          'merge2 keeps the agreed Kappa, Omega/Upsilon through a listed alternative, one Tau across the border; '
          'Alpha against Lambda is unpaired (got %s, unpaired %s, deduped %s)' % (chars2, unpaired2, deduped2))
    check(kap and kap[0]['box'] == [8025, 9025, 8010, 8810] and kap[0]['conf'] == 'clear' and tau and tau[0]['region'] == 5,
          'the kept box is the two readers\' mean, the conf the better one, and the border Tau is the copy centred in its own core (region 5)')
    m2path, l2path, r2path = (os.path.join(tmp, x) for x in ('p2-marks.json', 'p2-letters.json', 'p2-ref.json'))
    json.dump(runs2, open(m2path, 'w', encoding='utf-8'), ensure_ascii=False)
    rc, out = run(MERGE2, '--marks', m2path, '--out', l2path)
    rc2, out2 = run(RFL, '--letters', l2path, '--out', r2path) if rc == 0 else (1, out)
    R2r = json.load(open(r2path, encoding='utf-8')) if rc2 == 0 else {'letters': []}
    k2 = [L for L in R2r['letters'] if L['letter'] == 'Κ']
    check(rc2 == 0 and len(R2r['letters']) == 3 and k2 and (k2[0]['i'], k2[0]['j']) == (2130.75, 2102.0),
          'ref-from-letters reads the pass-2 file at pool 4: Kappa\'s map box [8025,9025)x[8010,8810) centres at render ((8025+9025)/2/4 - 0.5, (8010+8810)/2/4 - 0.5) = (2130.75, 2102.0) (got %s; %s)'
          % ([(L['letter'], L['i'], L['j']) for L in R2r['letters']], (out + out2)[-200:]))
    short = [r for r in runs2 if not (r['region']['k'] == 8 and r['reader'] == 'b')]
    json.dump(short, open(m2path, 'w', encoding='utf-8'), ensure_ascii=False)
    rc, out = run(MERGE2, '--marks', m2path, '--out', os.path.join(tmp, 'p2-short.json'))
    check(rc != 0 and 'region 8 reader b' in out, 'merge2 refuses a run missing a reader, writing nothing')

    # ---- the workflow on mock agents --------------------------------------------------------------------
    node = shutil.which('node')
    if node:
        r = subprocess.run([node, os.path.join(HERE, 'test_workflow_mock.mjs')], capture_output=True, text=True)
        try:
            res = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            res = dict(ok=False, checks=[], failures=[(r.stdout + r.stderr)[-400:]])
        for c in res['checks']:
            check(True, 'workflow: ' + c)
        for c in res['failures']:
            check(False, 'workflow: ' + c)
    else:
        check(False, 'workflow: node not found, the mock run did not happen')
finally:
    shutil.rmtree(tmp, ignore_errors=True)

n_fail = sum(1 for ok, _ in RESULTS if not ok)
print('\n%d checks, %d failed' % (len(RESULTS), n_fail))
sys.exit(1 if n_fail else 0)
