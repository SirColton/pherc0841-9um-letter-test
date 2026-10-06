"""note4-verdict.py - prereg-v2 note 4's letter-read verdict for 252405, computed, never typed.

  note4-verdict.py --pool pe [--pool pf ...] [--ceiling pd] [--out VERDICT.json]

Each pool is read from <pool>-calls.json and <pool>-key.json beside this file, scored against the locked
252405-ref.json by score.py (same rule, same radius), and then:
  GO needs ALL of (note 4):
    (a) matched >= 6 of 8 (note 2, m = 6);
    (b) <= 24 calls on the mirror phase nulls AND <= 24 on the depth-shuffled nulls, each scaled to 25 tiles
        (the clarification of 19:13:10: the cap applies to each null type separately);
    (c) precision >= 0.11.
  REPORTED, NOT DECIDING: NET, LOCATED, LOCATED-BOX, and the same score with only calls of confidence >= 0.40
  (tau, a rule of the ceiling: its highest null-call confidence 0.35 plus 0.05).
The ceiling pool is printed beside the others for the side-by-side note 4 asks for; it is never judged GO.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, '252405-ref.json')
BAR_MATCHED, NULL_CAP, PRECISION_MIN, TAU, PER = 6, 24, 0.11, 0.40, 25


def score(reads, key):
    fd, out = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'score.py'), 'score', '--reads', reads, '--key', key,
                        '--reference', REF, '--out', out], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit('score.py refused %s (exit %d):\n%s' % (reads, r.returncode, (r.stdout + r.stderr)[-1500:]))
    rep = json.load(open(out))
    os.unlink(out)
    return rep


def floored(reads):
    d = json.load(open(reads, encoding='utf-8'))
    keep = [c for c in d['calls'] if isinstance(c.get('confidence'), (int, float)) and c['confidence'] >= TAU]
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    json.dump(dict(d, calls=keep), open(path, 'w', encoding='utf-8'))
    return path


def judge(rep):
    by = rep['null_calls_by_kind']
    tiles = rep['tiles_read_by_kind']
    scaled = {k: by.get(k, 0) * PER / float(tiles[k]) for k in ('phase', 'depth') if tiles.get(k)}
    a = rep['matched'] >= BAR_MATCHED
    b = all(v <= NULL_CAP for v in scaled.values())
    c = rep['precision'] is not None and rep['precision'] >= PRECISION_MIN
    return dict(a_matched=a, b_nulls=b, c_precision=c, go=a and b and c, nulls_scaled_to_25=scaled)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', action='append', required=True)
    ap.add_argument('--ceiling')
    ap.add_argument('--out')
    A = ap.parse_args(argv)
    rows = []
    for name in ([A.ceiling] if A.ceiling else []) + A.pool:
        reads, key = os.path.join(HERE, name + '-calls.json'), os.path.join(HERE, name + '-key.json')
        rep = score(reads, key)
        fl = floored(reads)
        try:
            rep_t = score(fl, key)
        finally:
            os.unlink(fl)
        rows.append(dict(pool=name, ceiling=name == A.ceiling, map=json.load(open(key))['meta']['map'], full=rep, tau=rep_t,
                         verdict=None if name == A.ceiling else judge(rep)))
    for r in rows:
        f, t, v = r['full'], r['tau'], r['verdict']
        print('%s  %s' % (r['pool'], os.path.basename(r['map'])) + ('  (ceiling, not judged)' if r['ceiling'] else ''))
        print('  matched %d of %d   precision %s   nulls by kind %s on %s tiles   NET %d   LOCATED %d   LOCATED-BOX %d' % (
            f['matched'], f['n_ref'], 'n/a' if f['precision'] is None else '%.3f' % f['precision'], f['null_calls_by_kind'],
            {k: n for k, n in f['tiles_read_by_kind'].items() if k != 'real'}, f['net_letters'], f['located'], f['located_box']))
        print('  at tau %.2f: matched %d, nulls by kind %s' % (TAU, t['matched'], t['null_calls_by_kind']))
        if v:
            print('  (a) matched >= %d: %s   (b) each null type <= %d per 25 tiles %s: %s   (c) precision >= %.2f: %s' % (
                BAR_MATCHED, 'yes' if v['a_matched'] else 'NO', NULL_CAP,
                {k: round(x, 2) for k, x in v['nulls_scaled_to_25'].items()}, 'yes' if v['b_nulls'] else 'NO',
                PRECISION_MIN, 'yes' if v['c_precision'] else 'NO'))
            print('  LETTER READ: %s' % ('GO' if v['go'] else 'NO GO'))
    if A.out:
        if os.path.exists(A.out):
            raise SystemExit('REFUSED: %s exists; a verdict is written once' % A.out)
        with open(A.out, 'x', encoding='utf-8') as f:
            json.dump(dict(rule='prereg-v2 note 4 (with its 19:13:10 clarification)', bar_matched=BAR_MATCHED,
                           null_cap_per_type=NULL_CAP, precision_min=PRECISION_MIN, tau=TAU, reference=REF,
                           rows=[dict(pool=r['pool'], ceiling=r['ceiling'], map=r['map'], verdict=r['verdict'],
                                      matched=r['full']['matched'], precision=r['full']['precision'],
                                      null_calls_by_kind=r['full']['null_calls_by_kind'], net=r['full']['net_letters'],
                                      located=r['full']['located'], located_box=r['full']['located_box'],
                                      matches=r['full']['matches'], tau_matched=r['tau']['matched'],
                                      tau_null_calls_by_kind=r['tau']['null_calls_by_kind']) for r in rows]), f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
