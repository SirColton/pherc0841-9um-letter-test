"""student-train-w00bigproof.py - proves student-train.py's w00big check (amendment 1 item 1) can fail.
Copies the w00 render (stack, mask, spmap grid, grid.json) to geom/student-9um/patch/scratch/, changes ONE voxel
of the copy per case, rewrites only the copy's grid.json stack_sha1 so the harness reaches the check, and runs the
real harness (polite-run.py student-train.py --dry 1 --seg w00=<copy> --seg 552896=<real>) on it. The real render
files are opened read-only and never written. Each case restores its voxel; the copy is deleted at the end.
Cases (window = render rows [1900, 2895) cols [2500, 3745), interior = at least 5 px inside it):
  A  first interior px, window (5, 5), layer 0, |diff| 1          -> harness must STOP on the check (exit 2)
  B  last interior px, window (989, 1239), layer 40, |diff| 1     -> must STOP (exit 2)
  C  frame px just outside, window (4, 600), layer 20, |diff| 128 -> must PASS the check; frame max rises to 128
Usage (from <workdir>): python student-train-w00bigproof.py
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

import runfile

PY = sys.executable
REAL = 'geom/student-9um/renders/w00'
OUT = 'geom/student-9um/patch'
SCR = os.path.join(OUT, 'scratch')
CP = os.path.join(SCR, 'w00')
WIN = (1900, 2895, 2500, 3745)
REP = os.path.join(OUT, 'w00bigproof-run.txt')
runfile.claim(REP)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_ok():
    o = os.environ.get('LOAD_LINE', 'RAM 0% CPU 0%')  # the private load probe is not published; set LOAD_LINE to your own reading
    ram, cpu = (int(re.search(k + r' (\d+)%', o).group(1)) for k in ('RAM', 'CPU'))
    return ram < 80 and cpu < 80, o


say('student-train-w00bigproof.py: the w00big check (amendment 1 item 1) on a scratch copy with one voxel changed')
say('  harness student-train.py sha256 %s' % sha256('student-train.py'))
os.makedirs(SCR, exist_ok=True)
real_sha = {}
for suf in ('-stack.npy', '-mask.npy', '-spmap-grid.npy', '-grid.json'):
    real_sha[suf] = sha256(REAL + suf)
    shutil.copyfile(REAL + suf, CP + suf)
    if sha256(CP + suf) != real_sha[suf]:
        raise SystemExit('copy of %s differs' % suf)
say('  copied %s-{stack,mask,spmap-grid}.npy and -grid.json to %s-*; each copy sha256 equals its source' % (REAL, CP))
g0 = json.load(open(REAL + '-grid.json', encoding='utf-8'))

CASES = [('A', 5, 5, 0, 'xor1', 2, 'FAIL'),
         ('B', 989, 1239, 40, 'xor1', 2, 'FAIL'),
         ('C', 4, 600, 20, 'plus128', 0, 'PASS')]
results = []
for tag, wy, wx, layer, how, want_rc, want_check in CASES:
    y, x = WIN[0] + wy, WIN[2] + wx
    st = np.load(CP + '-stack.npy', mmap_mode='r+')
    v0 = int(st[layer, y, x])
    v1 = v0 ^ 1 if how == 'xor1' else (v0 + 128) % 256
    st[layer, y, x] = v1
    st.flush()
    s1 = hashlib.sha1(np.ascontiguousarray(st).tobytes()).hexdigest()[:12]
    del st
    g = dict(g0)
    g['stack_sha1'] = s1
    json.dump(g, open(CP + '-grid.json', 'w', encoding='utf-8'), indent=1)
    ok, lo = load_ok()
    say('')
    say('case %s: render (%d, %d) = window (%d, %d), layer %d: %d -> %d (|diff| %d vs w00big there, if the real voxel equals it); copy stack sha1 %s'
        % (tag, y, x, wy, wx, layer, v0, v1, abs(v1 - v0), s1))
    say('  load before: %s' % lo)
    if not ok:
        raise SystemExit('load at or over 80%; not launching')
    out = os.path.join(OUT, 'proof-case-%s' % tag)
    cmd = [PY, 'polite-run.py', 'student-train.py', '--dry', '1', '--seg', 'w00=' + CP.replace('\\', '/'),
           '--seg', '552896=geom/student-9um/renders/552896', '--out', out.replace('\\', '/')]
    say('  run: %s' % ' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)))
    p = subprocess.run(cmd, capture_output=True, text=True)
    chk = [l_ for l_ in p.stdout.splitlines() if 'CHECK w00 vs w00big' in l_ or l_.startswith('    frame,') or l_.startswith('STOP') or l_.startswith('DRY done')]
    for l_ in chk:
        say('  | ' + l_.strip())
    got_check = 'PASS' if any('CHECK w00 vs w00big' in l_ and l_.rstrip().endswith('PASS') for l_ in chk) else \
        ('FAIL' if any('CHECK w00 vs w00big' in l_ and l_.rstrip().endswith('FAIL') for l_ in chk) else 'ABSENT')
    good = p.returncode == want_rc and got_check == want_check
    say('  exit %d (want %d), check %s (want %s) -> %s' % (p.returncode, want_rc, got_check, want_check, 'AS PREDICTED' if good else 'NOT AS PREDICTED'))
    if p.stderr.strip():
        say('  stderr tail: ' + ' / '.join(p.stderr.strip().splitlines()[-3:]))
    results.append((tag, good))
    st = np.load(CP + '-stack.npy', mmap_mode='r+')
    st[layer, y, x] = v0
    st.flush()
    del st

say('')
shutil.copyfile(REAL + '-grid.json', CP + '-grid.json')
restored = all(sha256(CP + suf) == real_sha[suf] for suf in real_sha)
say('  copy restored to its source sha256, all four files: %s' % restored)
for suf in real_sha:
    if sha256(REAL + suf) != real_sha[suf]:
        raise SystemExit('the REAL %s changed during the proof' % suf)
say('  real w00 render files unchanged (sha256 re-read after the cases): True')
shutil.rmtree(SCR)
say('  scratch copy deleted: %s' % (not os.path.exists(SCR)))
allgood = all(gd for _, gd in results) and restored
say('PROOF %s: %s' % ('HOLDS' if allgood else 'BROKEN', ', '.join('%s %s' % (t, 'ok' if gd else 'WRONG') for t, gd in results)))
runfile.done(REP, '\n'.join(lines) + '\n')
sys.exit(0 if allgood else 1)
