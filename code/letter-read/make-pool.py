"""make-pool.py - the tile pool for the blind letter read: one 9 um map cut into tiles, plus NULL tiles
made from the same map, under opaque names, with the key written OUTSIDE the tiles folder.

WHAT A TILE IS. A square window of the map (default 8.0 mm = 832 px at 9.612 um), shown through one
fixed display transform that is the same for every tile, real or null:
  card     ink_9um's card recipe, clip((p/255 - 0.25) / 0.5) - c0841-9um.py's panel(); the default
           for a .npy map
  raw      clip(p, 0, 255) - for an 8-bit probability image such as Hecate's PNG; the default for one
  stretch  the real map's 1st-99th percentile inside the mask, computed once
Windows are spread evenly across the frame so neighbours overlap by at least --overlap-mm (default
4.0 mm): a letter up to that size lies whole inside at least one tile. WHY 8 mm: fl-tiles-l0.py found
on this same w00big frame that 6 mm tiles cut its letters in half and showed only blobs, so it went to
900 px (8.6 mm); the w00big frame is only 9.56 x 11.97 mm, so 8 mm with 4 mm overlap gives a 2 x 2 grid
(origins rows 0, 163; cols 0, 413) and a 4 mm letter is whole in at least one tile.

THE NULLS, all made from the same map, each from one window of the real grid (its 'origin' in the key)
and shown through the same transform, so tile size, brightness histogram and blob texture match:
  phase  per tile, the record's negative (fl-tiles-l0.py, prereg controls_on_252405): 2D FFT amplitude
         kept, phase replaced by a uniform random phase from a seeded generator (the phase of the FFT
         of white noise, so the field stays real), then rank-remapped onto the tile's own values so
         the histogram is identical, then --iaaft rounds (default 40) of re-imposing the amplitude and
         re-ranking, because one remap alone shifts the blob-size spectrum (--iaaft 0 is the record's
         one-shot version; --no-rank-remap keeps the amplitude exact and the histogram not).
  block  per tile, letter shape destroyed and texture kept: the tile is cut into ~1 mm blocks
         (--block-mm; the record asked for ~100 px), the blocks are permuted, and seams are
         feathered over --feather-mm because hard seams were as obvious to a reader as the old strip
         shuffle (fl-tiles-l0.py); then the same rank remap.
  depth  NOT MADE BY THIS SCRIPT. A depth-shuffled null is the model's output on a depth-permuted input
         stack (prereg: default_rng(0).permutation(21) before the rev flip), so it needs a model run,
         which this script must not do (no GPU, no re-run). No such w00big map is on disk. When one is
         made with the same checkpoint and recipe, pass it as --depth-map and it is
         cut on the same grid as kind 'depth'.
By default the null tiles number as many as the real tiles, split evenly over the null kinds, so the
literal NET LETTERS = matched - null calls compares equal tile counts (score.py also prints the
area-matched form when they differ).

BLINDING. Every tile gets a shuffled opaque id ('t' + 7 hex digits from the seeded generator), so the
name, the alphabetical order and the write order say nothing about the kind. The PNG holds pixels only.
The key (tile -> kind, origin_i, origin_j, size_px, ...) is refused anywhere inside the tiles folder.
Deterministic: the same map, mask, arguments and --seed give byte-identical tiles and key.

Usage: make-pool.py --map M.npy|M.png --tiles-dir DIR --key KEY.json --seed N
                    [--mask MASK.npy] [--grid GRID.json] [--pitch-um 9.612]
                    [--tile-mm 8.0] [--overlap-mm 4.0] [--display auto|card|raw|stretch]
                    [--phase N] [--block N] [--depth-map D.npy [--depth N]]
                    [--block-mm 1.0] [--feather-mm 0.12] [--iaaft 40] [--no-rank-remap] [--upscale 1]
CPU numpy/PIL only. Writes the tiles folder (must be new or empty) and the key (must not exist).
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
NULL_KINDS = ('phase', 'depth', 'block')     # kind order for null assignment, fixed


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_map(path):
    """A 2D map as float64 plus the name of its source dtype."""
    if path.lower().endswith('.npy'):
        a = np.load(path)
    else:
        a = np.asarray(Image.open(path))
    if a.ndim != 2:
        raise SystemExit('REFUSED: %s is %s, not a 2D map' % (path, a.shape))
    return a.astype(np.float64), str(a.dtype)


def positions(L, T, O):
    """Evenly spread window origins along one axis: first at 0, last flush with the end, neighbours
    overlapping by at least O. A frame shorter than a tile gives one window of the frame's length."""
    if L <= T:
        return [0], L
    n = int(math.ceil((L - T) / float(T - O))) + 1
    return [int(round(k * (L - T) / float(n - 1))) for k in range(n)], T


def display(a, how, lo=None, hi=None):
    if how == 'card':
        v = np.clip((a / 255.0 - 0.25) / 0.5, 0, 1) * 255
    elif how == 'raw':
        v = np.clip(a, 0, 255)
    elif how == 'stretch':
        v = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255
    else:
        raise ValueError(how)
    return np.rint(v).astype(np.uint8)


def rank_remap(q, a):
    """Values of a, placed in the rank order of q: a's histogram exactly, q's arrangement."""
    out = np.empty(a.size, np.float64)
    out[np.argsort(q, axis=None, kind='stable')] = np.sort(a, axis=None, kind='stable')
    return out.reshape(a.shape)


def phase_scramble(a, rng, remap=True, iters=40, mirror=False):
    """FFT amplitude of a kept, phase uniform random (the phase of white noise's FFT, Hermitian, so
    the result is real); the DC phase is a's own, so the mean is kept. remap: rank-remap onto a's
    values, then `iters` IAAFT rounds (amplitude imposed, values re-ranked), ending on the values, so
    the histogram is exact and the spectrum close. On a bimodal synthetic tile one remap alone moved the
    radially averaged power 0.37 decades on average; 40 rounds bring it to 0.07 (test_letter_read.py)."""
    if mirror:
        # The FFT sees the tile as periodic, and its unmatched edges put a cross of power on the two
        # frequency axes that random phase turns into full-length streaks readers named (a reviewer,
        # 2026-10-05). The mirror-symmetric 2H x 2W extension has no edge jump; scramble it,
        # keep the first H x W, and rank-remap that onto a so the histogram stays exact.
        H, W = a.shape
        q = phase_scramble(np.block([[a, a[:, ::-1]], [a[::-1], a[::-1, ::-1]]]), rng, remap, iters)[:H, :W]
        return rank_remap(q, a) if remap else q
    F = np.fft.fft2(a)
    amp = np.abs(F)
    phi = np.angle(np.fft.fft2(rng.standard_normal(a.shape)))
    phi[0, 0] = np.angle(F[0, 0])
    q = np.real(np.fft.ifft2(amp * np.exp(1j * phi)))
    if not remap:
        return q
    x = rank_remap(q, a)
    for _ in range(iters):
        x = rank_remap(np.real(np.fft.ifft2(amp * np.exp(1j * np.angle(np.fft.fft2(x))))), a)
    return x


def ramp(b, m):
    """1D weight over a block extended by m on each side: linear ramps of width 2m centred on the
    block edges, flat 1 between, so neighbouring blocks' weights sum to exactly 1 across a seam."""
    u = np.arange(-m, b + m, dtype=np.float64)
    if m == 0:
        return np.ones(b)
    w = np.ones_like(u)
    up = u < m
    w[up] = (u[up] + m + 0.5) / (2 * m)
    dn = u >= b - m
    w[dn] = (b + m - u[dn] - 0.5) / (2 * m)
    return np.clip(w, 0, 1)


def block_shuffle(a, rng, b, m, remap=True):
    """a cut into b x b blocks, the blocks permuted, seams feathered over 2m px (m = 0: hard seams,
    values exactly permuted when b divides the tile)."""
    if b < 2 * m:
        raise SystemExit('REFUSED: block %d px is narrower than twice the feather %d px' % (b, m))
    H, W = a.shape
    ny, nx = int(math.ceil(H / float(b))), int(math.ceil(W / float(b)))
    P = np.pad(a, ((m, ny * b - H + m), (m, nx * b - W + m)), mode='symmetric')
    w2 = np.outer(ramp(b, m), ramp(b, m))
    out = np.zeros(P.shape)
    wt = np.zeros(P.shape)
    perm = rng.permutation(ny * nx)
    e = b + 2 * m
    for k in range(ny * nx):
        dy, dx = divmod(k, nx)
        sy, sx = divmod(int(perm[k]), nx)
        out[dy * b:dy * b + e, dx * b:dx * b + e] += w2 * P[sy * b:sy * b + e, sx * b:sx * b + e]
        wt[dy * b:dy * b + e, dx * b:dx * b + e] += w2
    q = (out / np.maximum(wt, 1e-12))[m:m + H, m:m + W]
    return rank_remap(q, a) if remap else q


def inside(child, parent):
    c, p = os.path.normcase(os.path.abspath(child)), os.path.normcase(os.path.abspath(parent))
    try:
        return os.path.commonpath([c, p]) == p
    except ValueError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--map', required=True)
    ap.add_argument('--mask')
    ap.add_argument('--grid', help='the render grid json (pitch_um_in_plane, H, W are checked)')
    ap.add_argument('--pitch-um', type=float)
    ap.add_argument('--tiles-dir', required=True)
    ap.add_argument('--key', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tile-mm', type=float, default=8.0)
    ap.add_argument('--overlap-mm', type=float, default=4.0)
    ap.add_argument('--display', default='auto', choices=('auto', 'card', 'raw', 'stretch'))
    ap.add_argument('--phase', type=int)
    ap.add_argument('--block', type=int)
    ap.add_argument('--depth', type=int)
    ap.add_argument('--depth-map')
    ap.add_argument('--block-mm', type=float, default=1.0)
    ap.add_argument('--feather-mm', type=float, default=0.12)
    ap.add_argument('--no-rank-remap', action='store_true')
    ap.add_argument('--iaaft', type=int, default=40, help='IAAFT rounds after the phase null\'s rank remap')
    ap.add_argument('--upscale', type=int, default=1)
    ap.add_argument('--windows-from', help='a score.py reference: keep only the real windows that hold at least one of '
                    'its letters whole (a fixed development subset; readers never see the reference)')
    ap.add_argument('--empty-windows', type=int, help='with --windows-from: this many real windows that touch NO reference '
                    'box, at least half valid, seeded (default: as many as the letter windows)')
    ap.add_argument('--no-mirror', action='store_true',
                    help='phase nulls scramble the tile as periodic, as the w00big pools before 2026-10-05 did; the '
                         'default scrambles the mirror-symmetric extension (prereg-v2 note 3)')
    A = ap.parse_args(argv)
    A.mirror = not A.no_mirror

    # ---- refusals first: nothing is written unless every one passes ----------------------------
    if inside(A.key, A.tiles_dir):
        raise SystemExit('REFUSED: the key %s is inside the tiles folder %s; a reader pointed at the '
                         'tiles must never be able to reach it' % (A.key, A.tiles_dir))
    if os.path.exists(A.key):
        raise SystemExit('REFUSED: %s exists; a key is never overwritten (delete it by hand to remake '
                         'the pool)' % A.key)
    if os.path.isdir(A.tiles_dir) and os.listdir(A.tiles_dir):
        raise SystemExit('REFUSED: %s is not empty; a pool is never mixed with another' % A.tiles_dir)
    if A.upscale < 1:
        raise SystemExit('REFUSED: --upscale must be >= 1')

    a, src_dtype = load_map(A.map)
    H, W = a.shape
    if A.mask:
        mask = np.load(A.mask).astype(bool)
        if mask.shape != a.shape:
            raise SystemExit('REFUSED: mask %s vs map %s' % (mask.shape, a.shape))
    else:
        mask = np.ones(a.shape, bool)
    pitch = A.pitch_um
    grid = None
    if A.grid:
        grid = json.load(open(A.grid, encoding='utf-8'))
        if (grid.get('H'), grid.get('W')) != (H, W):
            raise SystemExit('REFUSED: map %dx%d is not on the grid %s (%sx%s)' % (H, W, A.grid, grid.get('H'), grid.get('W')))
        gp = float(grid['pitch_um_in_plane'])
        if pitch is not None and abs(pitch - gp) > 1e-6:
            raise SystemExit('REFUSED: --pitch-um %g contradicts the grid %g' % (pitch, gp))
        pitch = gp
    if pitch is None:
        raise SystemExit('REFUSED: give --grid or --pitch-um')
    dmap = None
    if A.depth_map:
        dmap, _ = load_map(A.depth_map)
        if dmap.shape != a.shape:
            raise SystemExit('REFUSED: depth map %s vs map %s' % (dmap.shape, a.shape))
    elif A.depth:
        raise SystemExit('REFUSED: --depth needs --depth-map (this script never runs a model)')

    # A float map in 0-1 is a probability on the other scale: the card recipe assumes 0-255 and clipped
    # ScrollPrize's 2.4 um grid (0.11-0.96) to all black, a blank pool readers then read (2026-10-05).
    unit_scaled = src_dtype != 'uint8' and float(np.nanmax(a)) <= 1.0 + 1e-6
    if unit_scaled:
        a = a * 255.0
    how = A.display
    if how == 'auto':
        how = 'raw' if src_dtype == 'uint8' else 'card'
    lo = hi = None
    if how == 'stretch':
        lo, hi = (float(np.percentile(a[mask], p)) for p in (1, 99))
    fill = float(a[mask].mean())

    T = int(round(A.tile_mm * 1000 / pitch))
    O = int(math.ceil(A.overlap_mm * 1000 / pitch))
    if O >= T:
        raise SystemExit('REFUSED: overlap %d px >= tile %d px' % (O, T))
    rows, th = positions(H, T, O)
    cols, tw = positions(W, T, O)
    grid_pos = [(i, j) for i in rows for j in cols]
    all_pos = list(grid_pos)
    empty_pos = []
    n_grid = len(grid_pos)
    uncovered = []
    if A.windows_from:
        # A development subset, chosen once by geometry from the reference's boxes (box_px is [i_min, i_max, j_min,
        # j_max], inclusive): every window holding a letter whole. Readers never see the reference or the choice.
        boxes = [(L.get('id'), L['box_px']) for L in json.load(open(A.windows_from, encoding='utf-8'))['letters'] if L.get('box_px')]
        holds = lambda p, b: p[0] <= b[0] and b[1] < p[0] + th and p[1] <= b[2] and b[3] < p[1] + tw
        grid_pos = [p for p in grid_pos if any(holds(p, b) for _, b in boxes)]
        uncovered = [lid for lid, b in boxes if not any(holds(p, b) for p in grid_pos)]
        if not grid_pos:
            raise SystemExit('REFUSED: no window holds any letter of %s whole' % A.windows_from)
        # And as many EMPTY real windows, touching no box of the reference (counted or not) and at least half valid,
        # so a variant that calls more letters everywhere shows false calls on real tiles too (a reviewer, 2026-10-06).
        refd = json.load(open(A.windows_from, encoding='utf-8'))
        allb = [L['box_px'] for L in refd['letters'] + refd.get('not_scored', []) if L.get('box_px')]
        touch = lambda p, b: p[0] <= b[1] and b[0] < p[0] + th and p[1] <= b[3] and b[2] < p[1] + tw
        letter_set = set(grid_pos)
        eligible = [p for p in all_pos if p not in letter_set and not any(touch(p, b) for b in allb)
                    and mask[p[0]:p[0] + th, p[1]:p[1] + tw].mean() >= 0.5]
        want = len(grid_pos) if A.empty_windows is None else A.empty_windows
        pick = np.random.default_rng([A.seed, 3]).permutation(len(eligible))[:want]
        empty_pos = sorted(eligible[int(x)] for x in pick)
        grid_pos = sorted(letter_set) + empty_pos
    n_real = len(grid_pos)
    min_ov = min([th - (rows[k + 1] - rows[k]) for k in range(len(rows) - 1)] +
                 [tw - (cols[k + 1] - cols[k]) for k in range(len(cols) - 1)] or [min(th, tw)])

    # Block nulls are off unless --block asks for them: on the first dry run a reader named them by
    # their seams ("a patchwork of ~100-104 px patches"), and a null a reader can recognise undercounts
    # false calls. I retired them until the feather hides the seams (2026-10-05).
    kinds = [k for k in NULL_KINDS if (k != 'depth' or dmap is not None) and (k != 'block' or A.block)]
    given = {'phase': A.phase, 'block': A.block, 'depth': A.depth}
    counts = {}
    if all(given[k] is None for k in kinds):
        base, extra = divmod(n_real, len(kinds))
        for x, k in enumerate(kinds):
            counts[k] = base + (1 if x < extra else 0)
    else:
        for k in kinds:
            counts[k] = given[k] if given[k] is not None else 0
    if any(v < 0 for v in counts.values()):
        raise SystemExit('REFUSED: negative null count')

    # null sources: one stream of seeded permutations of the real grid, dealt to the kinds in order,
    # so with the default counts every grid window gives exactly one null
    src_rng = np.random.default_rng([A.seed, 1])
    stream = []
    need = sum(counts.values())
    while len(stream) < need:
        stream.extend(int(x) for x in src_rng.permutation(n_real))
    tiles = [dict(kind='real', origin=grid_pos[k], k=k) for k in range(n_real)]
    s = 0
    for kind in kinds:
        for k in range(counts[kind]):
            tiles.append(dict(kind=kind, origin=grid_pos[stream[s]], k=k))
            s += 1

    # opaque names: unique, drawn from their own stream, assigned in a shuffled order
    name_rng = np.random.default_rng([A.seed, 2])
    names = set()
    while len(names) < len(tiles):
        names.add('t%07x' % int(name_rng.integers(0, 1 << 28)))
    names = sorted(names)
    order = name_rng.permutation(len(tiles))
    for t, n in zip((tiles[int(x)] for x in order), names):
        t['name'] = n

    b = int(round(A.block_mm * 1000 / pitch))
    m = int(round(A.feather_mm * 1000 / pitch))
    remap = not A.no_rank_remap
    os.makedirs(A.tiles_dir, exist_ok=True)
    key_tiles = {}
    pending = []
    for t in sorted(tiles, key=lambda t: t['name']):
        i0, j0 = t['origin']
        win = (slice(i0, i0 + th), slice(j0, j0 + tw))
        tm = mask[win]
        src = (dmap if t['kind'] == 'depth' else a)[win].copy()
        src[~tm] = fill
        entry = dict(kind=t['kind'], origin_i=i0, origin_j=j0, size_px=[th, tw], scale=A.upscale,
                     subset=('empty' if (i0, j0) in empty_pos else 'letter') if A.windows_from else None,
                     png_px=[th * A.upscale, tw * A.upscale], source_box=[i0, j0, i0 + th, j0 + tw],
                     masked_frac=round(float(1 - tm.mean()), 4))
        if t['kind'] == 'phase':
            ss = [A.seed, 10, t['k']]
            v = phase_scramble(src, np.random.default_rng(ss), remap, A.iaaft, A.mirror)
            entry.update(null_seed=ss, rank_remap=remap, iaaft=A.iaaft if remap else 0, mirror=bool(A.mirror))
        elif t['kind'] == 'block':
            ss = [A.seed, 30, t['k']]
            v = block_shuffle(src, np.random.default_rng(ss), b, m, remap)
            entry.update(null_seed=ss, block_px=b, feather_px=m, rank_remap=remap)
        else:
            v = src
        if t['kind'] in ('phase', 'block') and remap and not tm.all():
            # The scramble ran over the fill in the masked pixels too, so some fill came back inside the
            # valid area as flat grey its real twin lacks (a tell; a reviewer, 2026-10-05). Its
            # valid pixels take the real window's valid values instead, in the scramble's rank order.
            v = v.copy()
            v[tm] = rank_remap(v[tm], src[tm])
        u8 = display(v, how, lo, hi)
        u8[~tm] = 0
        if t['kind'] == 'real' and tm.any() and int(u8[tm].max()) == int(u8[tm].min()):
            raise SystemExit('REFUSED: real tile at %s is one flat value (%d) after the %s display: the transform '
                             'wiped the map, so nothing was written' % (t['origin'], int(u8[tm].max()), how))
        im = Image.fromarray(u8)
        if A.upscale > 1:
            im = im.resize((tw * A.upscale, th * A.upscale), Image.NEAREST)
        pending.append((t['name'], im))
        key_tiles[t['name']] = entry
    for name, im in pending:
        im.save(os.path.join(A.tiles_dir, name + '.png'))

    by_kind = {}
    for e in key_tiles.values():
        by_kind[e['kind']] = by_kind.get(e['kind'], 0) + 1
    meta = dict(
        map=os.path.abspath(A.map), map_sha256=sha256(A.map), map_shape=[H, W], map_dtype=src_dtype,
        unit_scaled=bool(unit_scaled), mask=os.path.abspath(A.mask) if A.mask else None, mask_sha256=sha256(A.mask) if A.mask else None,
        grid=os.path.abspath(A.grid) if A.grid else None, pitch_um=pitch,
        depth_map=os.path.abspath(A.depth_map) if A.depth_map else None,
        depth_map_sha256=sha256(A.depth_map) if A.depth_map else None,
        windows_from=os.path.abspath(A.windows_from) if A.windows_from else None,
        windows_from_sha256=sha256(A.windows_from) if A.windows_from else None,
        windows_of_grid=[n_real, n_grid], letters_in_no_window=uncovered, empty_windows=len(empty_pos),
        depth_note=None if A.depth_map else
        'no depth-shuffled null: it needs a model run on a depth-permuted stack, which this script never does; '
        + ('block-shuffled null used as the second shape-destroying null' if A.block
           else 'phase nulls only (block nulls are off unless --block asks)'),
        tiles_dir=os.path.abspath(A.tiles_dir), seed=A.seed, tile_px=[th, tw], overlap_px_min=int(min_ov),
        overlap_mm_min=round(min_ov * pitch / 1000, 3), row_origins=rows, col_origins=cols,
        display=how, display_lohi=[lo, hi] if how == 'stretch' else None, upscale=A.upscale, mirror=bool(A.mirror),
        null_counts=counts, block_px=b, feather_px=m, rank_remap=remap, iaaft=A.iaaft if remap else 0, counts=by_kind,
        script_sha256=sha256(os.path.abspath(__file__)),
        box_convention='a reader box [x0, y0, x1, y1] is in png px; map px = (png + 0.5) / scale - 0.5; '
                       'w00big (i, j) = (origin_i + map y, origin_j + map x)')
    os.makedirs(os.path.dirname(os.path.abspath(A.key)), exist_ok=True)
    with open(A.key, 'w', encoding='utf-8') as f:
        json.dump(dict(meta=meta, tiles=key_tiles), f, indent=1, sort_keys=True)
    print('make-pool: map %s (%dx%d %s, sha256 %s), display %s, pitch %.3f um' % (
        A.map, H, W, src_dtype, meta['map_sha256'][:16], how, pitch))
    print('  tile %dx%d px (%.2f x %.2f mm), origins rows %s cols %s, min overlap %d px (%.2f mm)' % (
        th, tw, th * pitch / 1000, tw * pitch / 1000, rows, cols, min_ov, min_ov * pitch / 1000))
    if A.windows_from:
        print('  windows: %d of the %d on the grid hold a letter of %s whole, plus %d empty ones%s' % (
            n_real - len(empty_pos), n_grid, os.path.basename(A.windows_from), len(empty_pos),
            ('; in no window: ' + ', '.join(map(str, uncovered))) if uncovered else ''))
    print('  tiles: %s; block %d px, feather %d px, rank remap %s' % (
        ', '.join('%s %d' % (k, by_kind.get(k, 0)) for k in ('real',) + NULL_KINDS), b, m, remap))
    if meta['depth_note']:
        print('  ' + meta['depth_note'])
    print('  wrote %d tiles to %s and the key to %s' % (len(key_tiles), A.tiles_dir, A.key))
    return 0


if __name__ == '__main__':
    sys.exit(main())
