"""score.py - score a blind letter read against the LOCKED reference.

  score.py lock     --reference REF.json [--lock REF.json.sha256]
        writes the reference's sha256 once; refuses if a lock already exists. Lock BEFORE any read.
        Without --lock every command looks for REF.json.sha256, then REF.lock.json ({"sha256": ...},
        the form the w00big reference maker wrote).
  score.py manifest --key KEY.json --tiles-dir DIR [--reference REF.json] [--lock L]
        what the letter-read workflow's parent agent runs (readers never do): checks the key sits
        outside the tiles folder, every key tile has its PNG and nothing else is there, PNG sizes
        match, and (with --reference) the lock holds; prints JSON {ok, problems, pitch_um, tiles:
        [{tile, control, box, png_px}]}. Exit 5 when not ok.
  score.py count    --reads OUT.json
        prints {n_calls, box_sum, n_reads}: the workflow's writer checks its file with this.
  score.py score    --reads OUT.json --key KEY.json --reference REF.json [--lock L]
                    [--radius-mm 0.5] [--pitch-um P] [--out REPORT.json] [--allow-unread]

THE MATCH RULE (prereg letters.match_rule): a call matches when it names the same letter and its box
centre, mapped back to w00big render pixels through the key, lies within 0.5 mm of a reference letter's
(i, j); one-to-one, greedy by distance, each reference letter matched at most once. "The same letter"
includes a single letter the reference lists under alt (prereg v2 letter_read.match: "Reader A's listed
alternatives accepted"); an alt naming more than one letter ("I + T") is a reading of two marks, never
accepted, and is said in a NOTE. Box centre in png
px -> map px (c + 0.5) / scale - 0.5 -> w00big (origin_i + y, origin_j + x).
Real tiles overlap, so one letter can be called from several tiles: same-letter calls on REAL tiles
within the match radius of each other are merged first (highest confidence kept). A null tile is its own
world, so its calls merge only within that tile.
REPORTED: matched, precision = matched / real calls (merged), recall, calls on null tiles by kind, and
NET LETTERS = matched - null calls. When the pool's null tiles do not number the real tiles, the
area-matched form matched - null calls x real tiles / null tiles is printed beside it.
REFUSES (exit 4) when the reference's sha256 is not the one in its lock file, (exit 3) when a call names
a tile not in the key, and (exit 6) when a key tile has no read unless --allow-unread.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata

GREEK = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda',
         'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega')
_CHARS = dict(zip('αβγδεζηθικλμνξοπρστυφχψω', GREEK))
_CHARS.update({'ς': 'sigma', 'ϲ': 'sigma', 'ϐ': 'beta', 'ϵ': 'epsilon', 'ϑ': 'theta', 'ϕ': 'phi',
               'ϰ': 'kappa', 'ϱ': 'rho', 'ϖ': 'pi'})
# Latin capitals that are the same glyph as a Greek capital (C = lunate sigma); applied case-blind to a
# single character only
_LATIN = dict(zip('abezhikmnoptyxc', ('alpha', 'beta', 'epsilon', 'zeta', 'eta', 'iota', 'kappa', 'mu', 'nu',
                                      'omicron', 'rho', 'tau', 'upsilon', 'chi', 'sigma')))
_NAMES = {g: g for g in GREEK}
_NAMES.update({'lamda': 'lambda', 'lamba': 'lambda', 'omikron': 'omicron', 'ypsilon': 'upsilon',
               'khi': 'chi', 'ksi': 'xi', 'ro': 'rho', 'lunate sigma': 'sigma', 'lunate_sigma': 'sigma',
               'final sigma': 'sigma', 'sigma lunate': 'sigma', 'digamma': 'digamma'})
EXIT_TILE, EXIT_LOCK, EXIT_MANIFEST, EXIT_UNREAD = 3, 4, 5, 6


def norm_letter(x):
    """Canonical lowercase English name of a Greek letter, or None when it cannot be read as one."""
    if x is None:
        return None
    s = unicodedata.normalize('NFD', str(x).strip())
    s = ''.join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()
    if not s:
        return None
    if s in _NAMES:
        return _NAMES[s]
    if len(s) == 1:
        return _CHARS.get(s) or _LATIN.get(s)
    s2 = re.sub(r'[\s_\-]+', ' ', s)
    if s2 in _NAMES:
        return _NAMES[s2]
    for tok in re.findall(r'[a-z]+|[Ͱ-Ͽ]', s):
        if tok in _NAMES:
            return _NAMES[tok]
        if len(tok) == 1 and tok in _CHARS:
            return _CHARS[tok]
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def lock_candidates(ref):
    """Where a reference's lock may sit: <ref>.sha256 (this script's) or <ref stem>.lock.json (the form
    the reference maker wrote for w00big-ref.json, {"sha256": ...})."""
    stem = ref[:-5] if ref.lower().endswith('.json') else ref
    return [ref + '.sha256', stem + '.lock.json']


def lock_path(ref, lock):
    if lock:
        return lock
    for p in lock_candidates(ref):
        if os.path.exists(p):
            return p
    return lock_candidates(ref)[0]


def read_lock(lp):
    """The sha256 a lock file holds: a JSON object's "sha256", else the first token (sha256sum form)."""
    text = open(lp, encoding='utf-8').read()
    try:
        d = json.loads(text)
        if isinstance(d, dict) and isinstance(d.get('sha256'), str):
            return d['sha256'].strip().lower()
    except ValueError:
        pass
    toks = text.split()
    return toks[0].lower() if toks else ''


def check_lock(ref, lock):
    """None when the reference matches its lock, else the refusal text."""
    lp = lock_path(ref, lock)
    if not os.path.exists(ref):
        return 'no reference at %s' % ref
    if not os.path.exists(lp):
        return 'no lock file (looked for %s): lock the reference (score.py lock) before any read is scored' % (
            lock or ' or '.join(lock_candidates(ref)))
    want = read_lock(lp)
    got = sha256(ref)
    if got != want:
        return 'the reference %s (sha256 %s) is not the locked one (%s in %s)' % (ref, got, want or 'empty', lp)
    return None


def inside(child, parent):
    c, p = os.path.normcase(os.path.abspath(child)), os.path.normcase(os.path.abspath(parent))
    try:
        return os.path.commonpath([c, p]) == p
    except ValueError:
        return False


def load_key(path):
    d = json.load(open(path, encoding='utf-8'))
    tiles = d['tiles'] if isinstance(d, dict) and 'tiles' in d else d
    meta = d.get('meta', {}) if isinstance(d, dict) else {}
    out = {}
    for name, e in tiles.items():
        sz = e['size_px']
        h, w = (int(sz), int(sz)) if isinstance(sz, (int, float)) else (int(sz[0]), int(sz[1]))
        out[name] = dict(kind=e['kind'], oi=float(e['origin_i']), oj=float(e['origin_j']), h=h, w=w,
                         scale=float(e.get('scale', 1)), subset=e.get('subset'))
    return out, meta


def load_reads(path):
    d = json.load(open(path, encoding='utf-8'))
    if isinstance(d, list):
        return d, None, {}
    return d.get('calls', []), d.get('tiles_read'), d


def load_ref(path, pitch):
    d = json.load(open(path, encoding='utf-8'))
    letters = d['letters'] if isinstance(d, dict) else d
    refs, notes = [], []
    mm_mismatch = 0
    for k, L in enumerate(letters):
        name = norm_letter(L.get('letter'))
        if L.get('i') is not None and L.get('j') is not None:
            i, j = float(L['i']), float(L['j'])
            if L.get('x_mm') is not None and L.get('y_mm') is not None:
                if abs(float(L['x_mm']) - j * pitch / 1000) > 0.1 or abs(float(L['y_mm']) - i * pitch / 1000) > 0.1:
                    mm_mismatch += 1
        elif L.get('x_mm') is not None and L.get('y_mm') is not None:
            i, j = float(L['y_mm']) * 1000 / pitch, float(L['x_mm']) * 1000 / pitch
            notes.append('reference letter %s has no (i, j); placed from (y_mm, x_mm)' % L.get('id', k))
        else:
            notes.append('reference letter %s has no position; skipped' % L.get('id', k))
            continue
        # box_px is [i_min, i_max, j_min, j_max], inclusive (the w00big reference's own coords note). A box
        # that does not hold its own centre is in another order or wrong, and LOCATED-BOX skips it, said aloud.
        box = L.get('box_px')
        if box is not None:
            ok = isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)
            if not (ok and box[0] <= i <= box[1] and box[2] <= j <= box[3]):
                notes.append('reference letter %s: box_px %s does not hold its centre (%g, %g) as '
                             '[i_min, i_max, j_min, j_max]; no LOCATED-BOX for it' % (L.get('id', k), box, i, j))
                box = None
        alts = []
        for x in L.get('alt') or []:
            if re.search(r'[+&,/]', str(x)) or len(re.findall('[Ͱ-Ͽ]', str(x))) > 1:
                notes.append('reference letter %s: alternative %r names more than one letter; not accepted' % (L.get('id', k), x))
                continue
            a = norm_letter(x)
            if a is None:
                notes.append('reference letter %s: alternative %r is not a Greek letter; not accepted' % (L.get('id', k), x))
            elif a != name and a not in alts:
                alts.append(a)
        refs.append(dict(id=L.get('id', k), letter=name, raw=L.get('letter'), i=i, j=j, conf=L.get('confidence'),
                         box=[float(v) for v in box] if box is not None else None, alts=alts))
    if mm_mismatch:
        notes.append('%d reference letters have x_mm/y_mm more than 0.1 mm from j, i x pitch; (i, j) used' % mm_mismatch)
    return refs, notes


def box_of(c):
    b = c.get('box')
    if isinstance(b, dict):
        b = [b.get('x0'), b.get('y0'), b.get('x1'), b.get('y1')]
    if not isinstance(b, (list, tuple)) or len(b) != 4 or any(v is None for v in b):
        return None
    x0, y0, x1, y1 = (float(v) for v in b)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def merge(calls, radius_px):
    """Same-letter calls within radius_px of a kept call fold into it; highest confidence kept first."""
    order = sorted(range(len(calls)), key=lambda k: (-(calls[k]['conf'] or 0.0), calls[k]['tile'], k))
    kept = []
    for k in order:
        c = calls[k]
        for q in kept:
            if q['letter'] == c['letter'] and (q['i'] - c['i']) ** 2 + (q['j'] - c['j']) ** 2 <= radius_px ** 2 + 1e-9:
                q['merged'].append(c['ref'])
                break
        else:
            kept.append(dict(c, merged=[]))
    return kept


def match(calls, refs, radius_px):
    """One-to-one greedy by distance; identity must agree and both must be named."""
    pairs = []
    for a, c in enumerate(calls):
        if c['letter'] is None:
            continue
        for b, r in enumerate(refs):
            if r['letter'] is None or (c['letter'] != r['letter'] and c['letter'] not in r.get('alts', ())):
                continue
            d2 = (c['i'] - r['i']) ** 2 + (c['j'] - r['j']) ** 2
            if d2 <= radius_px ** 2 + 1e-9:
                pairs.append((d2, a, b))
    pairs.sort()
    used_c, used_r, out = set(), set(), []
    for d2, a, b in pairs:
        if a in used_c or b in used_r:
            continue
        used_c.add(a)
        used_r.add(b)
        out.append((a, b, d2 ** 0.5))
    return out


def cmd_lock(A):
    lp = A.lock or lock_candidates(A.reference)[0]
    for p in ([A.lock] if A.lock else []) + lock_candidates(A.reference):
        if os.path.exists(p):
            print('REFUSED: %s already exists (sha256 %s); a lock is written once' % (p, read_lock(p)))
            return EXIT_LOCK
    h = sha256(A.reference)
    with open(lp, 'w', encoding='utf-8') as f:
        f.write('%s  %s\n' % (h, os.path.basename(A.reference)))
    print('locked %s: sha256 %s -> %s' % (A.reference, h, lp))
    return 0


def cmd_manifest(A):
    problems = []
    if inside(A.key, A.tiles_dir):
        problems.append('the key is inside the tiles folder')
    key, meta = load_key(A.key)
    have = sorted(f for f in os.listdir(A.tiles_dir)) if os.path.isdir(A.tiles_dir) else []
    pngs = {f[:-4] for f in have if f.lower().endswith('.png')}
    other = [f for f in have if not f.lower().endswith('.png')]
    if not os.path.isdir(A.tiles_dir):
        problems.append('no tiles folder %s' % A.tiles_dir)
    if other:
        problems.append('non-PNG entries in the tiles folder: %s' % ' '.join(other[:5]))
    missing = sorted(set(key) - pngs)
    extra = sorted(pngs - set(key))
    if missing:
        problems.append('%d key tiles have no PNG: %s' % (len(missing), ' '.join(missing[:5])))
    if extra:
        problems.append('%d PNGs are not in the key: %s' % (len(extra), ' '.join(extra[:5])))
    tiles = []
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for name in sorted(key):
        e = key[name]
        png = [int(round(e['h'] * e['scale'])), int(round(e['w'] * e['scale']))]
        if Image is not None and name in pngs:
            with Image.open(os.path.join(A.tiles_dir, name + '.png')) as im:
                if [im.height, im.width] != png:
                    problems.append('%s.png is %dx%d, the key says %dx%d' % (name, im.height, im.width, png[0], png[1]))
        tiles.append(dict(tile=name, control=e['kind'] != 'real',
                          box=[e['oi'], e['oj'], e['oi'] + e['h'], e['oj'] + e['w']], png_px=png))
    if A.reference:
        why = check_lock(A.reference, A.lock)
        if why:
            problems.append('REFERENCE NOT LOCKED: ' + why)
    out = dict(ok=not problems, problems=problems, pitch_um=meta.get('pitch_um'), tiles=tiles)
    print(json.dumps(out))
    return 0 if not problems else EXIT_MANIFEST


def cmd_count(A):
    calls, _, d = load_reads(A.reads)
    s = 0
    for c in calls:
        s += sum(int(round(float(v))) for v in (c.get('box') or []))
    print(json.dumps(dict(n_calls=len(calls), box_sum=s, n_reads=len(d.get('reads', [])) if isinstance(d, dict) else 0)))
    return 0


def cmd_score(A):
    why = check_lock(A.reference, A.lock)
    if why:
        print('REFUSED: ' + why)
        return EXIT_LOCK
    key, meta = load_key(A.key)
    pitch = A.pitch_um or meta.get('pitch_um')
    if not pitch:
        print('REFUSED: no pitch (key meta has none; pass --pitch-um)')
        return EXIT_MANIFEST
    R = A.radius_mm * 1000.0 / pitch
    calls_in, tiles_read, rd = load_reads(A.reads)
    refs, notes = load_ref(A.reference, pitch)

    unknown = sorted({c.get('tile') for c in calls_in if c.get('tile') not in key}, key=str)
    if unknown:
        print('REFUSED: calls name tiles that are not in the key: %s' % ' '.join(map(str, unknown[:10])))
        return EXIT_TILE
    if tiles_read is None:
        notes.append('the reads file has no tiles_read list: a tile with no calls cannot be told from an unread one')
        read_set = set(key)
    else:
        read_set = set(tiles_read)
        bad = sorted(read_set - set(key))
        if bad:
            print('REFUSED: tiles_read names tiles not in the key: %s' % ' '.join(bad[:10]))
            return EXIT_TILE
    unread = sorted(set(key) - read_set)
    if unread and not A.allow_unread:
        kinds = {}
        for t in unread:
            kinds[key[t]['kind']] = kinds.get(key[t]['kind'], 0) + 1
        print('REFUSED: %d key tiles have no read (%s): %s; an unread null would count as zero calls. '
              'Re-read them, or pass --allow-unread to score anyway' % (
                  len(unread), ', '.join('%s %d' % kv for kv in sorted(kinds.items())), ' '.join(unread[:10])))
        return EXIT_UNREAD

    real_calls, null_calls, invalid = [], {}, []
    for k, c in enumerate(calls_in):
        e = key[c['tile']]
        b = box_of(c)
        name = norm_letter(c.get('letter'))
        conf = c.get('confidence')
        conf = float(conf) if isinstance(conf, (int, float)) else None
        rec = dict(ref=k, tile=c['tile'], letter=name, raw=c.get('letter'), conf=conf, box=c.get('box'))
        why = []
        nan = float('nan')
        if b is None:
            why.append('no box')
            rec.update(i=nan, j=nan)
        else:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            tx, ty = (cx + 0.5) / e['scale'] - 0.5, (cy + 0.5) / e['scale'] - 0.5
            rec.update(i=e['oi'] + ty, j=e['oj'] + tx, at=[e['oi'] + ty, e['oj'] + tx])
            if not (-0.5 <= tx <= e['w'] - 0.5 and -0.5 <= ty <= e['h'] - 0.5):
                # a centre off its own tile is a coordinate error (a crop's or a scaled view's px): it
                # stays a call, and it can match nothing
                why.append('box centre outside the tile')
                rec.update(i=nan, j=nan)
        if name is None:
            why.append('letter not readable as a Greek letter: %r' % c.get('letter'))
        if why:
            invalid.append(dict(rec, why='; '.join(why)))
        if e['kind'] == 'real':
            real_calls.append(rec)
        else:
            null_calls.setdefault(c['tile'], []).append(rec)

    # an invalid call still counts as a call (it is a claim that cannot match); only the finite ones merge
    finite = [c for c in real_calls if c['i'] == c['i']]
    nonfinite = [dict(c, merged=[]) for c in real_calls if c['i'] != c['i']]
    kept = merge(finite, R) + nonfinite
    pairs = match(kept, refs, R)
    matched = len(pairs)
    n_calls = len(kept)
    # LOCATED, a diagnostic and never GO (my ruling, 2026-10-05): a reference letter,
    # named or "?", counts once if any finite real call of any identity has its centre within the radius
    # of the letter's centre. It shows a reader finding the ink while misnaming it, which NET cannot.
    located = [r['id'] for r in refs
               if any((c['i'] - r['i']) ** 2 + (c['j'] - r['j']) ** 2 <= R ** 2 + 1e-9 for c in finite)]
    # LOCATED-BOX, the second diagnostic (prereg-v2 note 1, 2026-10-05): the same, with "centre inside the
    # reference letter's box" for the radius, because a big letter's misnamed half (one letter read as another) lands
    # inside it and outside 0.5 mm. The box is inclusive pixels, so it reaches half a pixel past each edge.
    boxed = [r for r in refs if r['box'] is not None]
    located_box = [r['id'] for r in boxed
                   if any(r['box'][0] - 0.5 <= c['i'] <= r['box'][1] + 0.5 and r['box'][2] - 0.5 <= c['j'] <= r['box'][3] + 0.5
                          for c in finite)]
    null_kept = {}
    for t, cs in null_calls.items():
        null_kept[t] = merge([c for c in cs if c['i'] == c['i']], R) + [dict(c, merged=[]) for c in cs if c['i'] != c['i']]
    null_by_kind = {}
    null_raw_by_kind = {}
    for t, cs in null_kept.items():
        k = key[t]['kind']
        null_by_kind[k] = null_by_kind.get(k, 0) + len(cs)
        null_raw_by_kind[k] = null_raw_by_kind.get(k, 0) + len(null_calls[t])
    tiles_by_kind, read_by_kind = {}, {}
    for t, e in key.items():
        tiles_by_kind[e['kind']] = tiles_by_kind.get(e['kind'], 0) + 1
        if t in read_set:
            read_by_kind[e['kind']] = read_by_kind.get(e['kind'], 0) + 1
    n_null_calls = sum(null_by_kind.values())
    # a development pool's real windows that hold no reference letter: every call there is false, whatever it names
    empty_tiles = [t for t, e in key.items() if e.get('subset') == 'empty']
    calls_on_empty = sum(1 for c in kept if key[c['tile']].get('subset') == 'empty')
    n_real_tiles = read_by_kind.get('real', 0)
    n_null_tiles = sum(v for k, v in read_by_kind.items() if k != 'real')
    net = matched - n_null_calls
    net_area = matched - n_null_calls * n_real_tiles / float(n_null_tiles) if n_null_tiles else None
    named_refs = [r for r in refs if r['letter'] is not None]
    precision = matched / float(n_calls) if n_calls else None
    recall = matched / float(len(named_refs)) if named_refs else None

    per_letter = {}
    for r in named_refs:
        per_letter.setdefault(r['letter'], [0, 0, 0])[1] += 1
    for c in kept:
        if c['letter'] is not None:
            per_letter.setdefault(c['letter'], [0, 0, 0])[2] += 1
    for a, b, d in pairs:
        per_letter[refs[b]['letter']][0] += 1
    by_conf = {}
    for b, r in enumerate(refs):
        cf = r['conf']
        lab = cf if isinstance(cf, str) else ('none' if cf is None else ('>=0.8' if cf >= 0.8 else ('0.5-0.8' if cf >= 0.5 else '<0.5')))
        by_conf.setdefault(lab, [0, 0])[1] += 1
    for a, b, d in pairs:
        r = refs[b]
        cf = r['conf']
        lab = cf if isinstance(cf, str) else ('none' if cf is None else ('>=0.8' if cf >= 0.8 else ('0.5-0.8' if cf >= 0.5 else '<0.5')))
        by_conf[lab][0] += 1

    um = pitch / 1000.0
    print('score.py: reads %s' % A.reads)
    print('  key %s, reference %s (locked, sha256 %s)' % (A.key, A.reference, sha256(A.reference)[:16]))
    print('  pitch %.3f um, match radius %.2f mm = %.1f px; map %s' % (pitch, A.radius_mm, R, meta.get('map', '?')))
    print('  tiles read: %s' % ', '.join('%s %d/%d' % (k, read_by_kind.get(k, 0), v) for k, v in sorted(tiles_by_kind.items())))
    print('  reference letters %d named (%d not readable as a Greek letter)' % (len(named_refs), len(refs) - len(named_refs)))
    print('  real calls %d raw, %d after merging same-letter calls within %.2f mm across tiles; %d invalid' % (
        len(real_calls), n_calls, A.radius_mm, sum(1 for x in invalid if key[x['tile']]['kind'] == 'real')))
    print('  MATCHED %d   precision %s   recall %s' % (
        matched, 'n/a' if precision is None else '%.3f' % precision, 'n/a' if recall is None else '%.3f' % recall))
    print('  calls on null tiles: %d (%s; raw %s)' % (
        n_null_calls, ', '.join('%s %d on %d tiles' % (k, null_by_kind.get(k, 0), read_by_kind.get(k, 0))
                                for k in sorted(read_by_kind) if k != 'real') or 'no null tiles read',
        ', '.join('%s %d' % kv for kv in sorted(null_raw_by_kind.items())) or '0'))
    print('  NET LETTERS = matched - null calls = %d - %d = %d' % (matched, n_null_calls, net))
    if empty_tiles:
        print('  calls on the %d empty real windows (no reference letter): %d after merging' % (len(empty_tiles), calls_on_empty))
    print('  LOCATED %d of %d reference letters (any identity, centre within %.2f mm; diagnostic, never GO)%s' % (
        len(located), len(refs), A.radius_mm, (': ' + ', '.join(located)) if located else ''))
    print('  LOCATED-BOX %d of %d reference letters with a box (any identity, centre inside the letter\'s box; '
          'diagnostic, never GO)%s' % (len(located_box), len(boxed), (': ' + ', '.join(located_box)) if located_box else ''))
    if net_area is not None and n_null_tiles != n_real_tiles:
        print('  (null tiles %d != real tiles %d: area-matched NET = %d - %d x %d/%d = %.2f)' % (
            n_null_tiles, n_real_tiles, matched, n_null_calls, n_real_tiles, n_null_tiles, net_area))
    print('  per letter (matched / reference / called): %s' % ', '.join(
        '%s %d/%d/%d' % (k, v[0], v[1], v[2]) for k, v in sorted(per_letter.items())))
    print('  matched by reference confidence: %s' % ', '.join('%s %d/%d' % (k, v[0], v[1]) for k, v in sorted(by_conf.items(), key=str)))
    for a, b, d in sorted(pairs, key=lambda p: refs[p[1]]['i']):
        c, r = kept[a], refs[b]
        print('    match  ref %-6s %-8s at (%4.0f, %4.0f)  call %s %s at (%4.0f, %4.0f)  %.2f mm%s%s' % (
            r['id'], r['letter'], r['i'], r['j'], c['tile'], c['letter'], c['i'], c['j'], d * um,
            ('  (+%d merged)' % len(c['merged'])) if c['merged'] else '',
            ('  (a listed alternative)' if c['letter'] != r['letter'] else '')))
    for n in notes:
        print('  NOTE ' + n)
    for x in invalid:
        print('  INVALID %s %r: %s' % (x['tile'], x['raw'], x['why']))

    rep = dict(reads=os.path.abspath(A.reads), reads_sha256=sha256(A.reads), key=os.path.abspath(A.key),
               key_sha256=sha256(A.key), reference=os.path.abspath(A.reference), reference_sha256=sha256(A.reference),
               map=meta.get('map'), map_sha256=meta.get('map_sha256'), pitch_um=pitch, radius_mm=A.radius_mm,
               n_ref=len(named_refs), n_ref_unnamed=len(refs) - len(named_refs),
               real_calls_raw=len(real_calls), real_calls=n_calls, matched=matched, precision=precision, recall=recall,
               null_calls=n_null_calls, null_calls_by_kind=null_by_kind, null_calls_raw_by_kind=null_raw_by_kind,
               tiles_by_kind=tiles_by_kind, tiles_read_by_kind=read_by_kind, unread=unread,
               net_letters=net, net_letters_area_matched=net_area, per_letter=per_letter, matched_by_ref_confidence=by_conf,
               located=len(located), located_ids=located, located_of=len(refs),
               located_box=len(located_box), located_box_ids=located_box, located_box_of=len(boxed),
               empty_real_tiles=len(empty_tiles), calls_on_empty_real=calls_on_empty,
               matches=[dict(ref_id=refs[b]['id'], letter=refs[b]['letter'], ref_i=refs[b]['i'], ref_j=refs[b]['j'],
                             tile=kept[a]['tile'], call_i=kept[a]['i'], call_j=kept[a]['j'], dist_mm=d * um,
                             merged_calls=len(kept[a]['merged']), call_letter=kept[a]['letter'],
                             via_alt=kept[a]['letter'] != refs[b]['letter']) for a, b, d in pairs],
               invalid=[dict(tile=x['tile'], letter=x['raw'], why=x['why']) for x in invalid], notes=notes)
    if A.out:
        with open(A.out, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=1)
        print('  wrote %s' % A.out)
    return 0


def main(argv=None):
    # A Greek letter printed to a Windows pipe (cp1252) raised and the score exited 1 before writing its
    # report; escaped output keeps every line and every exit code.
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(errors='backslashreplace')
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description='blind letter read: lock, manifest, count, score')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('lock')
    p.add_argument('--reference', required=True)
    p.add_argument('--lock')
    p = sub.add_parser('manifest')
    p.add_argument('--key', required=True)
    p.add_argument('--tiles-dir', required=True)
    p.add_argument('--reference')
    p.add_argument('--lock')
    p = sub.add_parser('count')
    p.add_argument('--reads', required=True)
    p = sub.add_parser('score')
    p.add_argument('--reads', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--lock')
    p.add_argument('--radius-mm', type=float, default=0.5)
    p.add_argument('--pitch-um', type=float)
    p.add_argument('--out')
    p.add_argument('--allow-unread', action='store_true')
    A = ap.parse_args(argv)
    return dict(lock=cmd_lock, manifest=cmd_manifest, count=cmd_count, score=cmd_score)[A.cmd](A)


if __name__ == '__main__':
    sys.exit(main())
