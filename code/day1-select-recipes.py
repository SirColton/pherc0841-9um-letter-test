"""day1-select-recipes.py - build prereg-v2's recipes R1-R4 on the w00big frame (selection data only).

prereg-v2 (geom/day1-9um/prereg-v2.json, sha256 checked; a mismatch stops the run):
  R1  best single claim-eligible-or-on-disk member (five-centre mean)
  R2  rank-normalised mean of the three MIT community members
  R3  R2 with mirror TTA
  R4  R2 plus soup42 and soup43, rank-normalised mean
Members are rev-order five-centre means (-4,-2,0,+2,+4) of day1-ink9-maps.py maps. Rank normalisation is
day1-score-baselines.py's rnorm, copied with its pixel set M (ranks over the scored pixels / their count,
0.5 elsewhere), and its mean5 is copied the same way, so a recipe here is the scorer's ENS rank-mean of
these members. A CHECK map rebuilds the scorer's own "ENS rank-mean soup42+soup43" from the on-disk maps:
scored, it must reproduce score-all's row (AUCsup 0.790) or the copy is wrong.
R1 is mechanical, by the prereg's selection rule on the score JSONs given: highest AUCsup; within 0.01 of
the best, highest hp r at 192 um, blur 2, vs the ScrollPrize 2.4 um map. Eligible: the three MIT members
and the released s43-75k, soup42, soup43. Not eligible: s43-50k (diagnostic only), Scheirer (reference
arm only), Hecate 9.6 (never in a decision). This selects R1's member, never the recipe.
Each recipe is written as <R>-c+0-rev.npy only so the unmodified scorer's file pattern picks it up: the
"c+0" there is a file-name slot, not a depth centre.
CPU only. Usage: day1-select-recipes.py <score s48 json> [...] -- <score s192 json> [...]
Writes geom/day1-9um/select/recipes/*.npy and geom/day1-9um/select/recipes.txt.
"""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter  # noqa: F401  (kept beside the scorer's imports for the copied block)
from scipy.stats import rankdata

import runfile

D9 = 'geom/day1-9um'
PREREG_SHA = '49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c'
if hashlib.sha256(open(os.path.join(D9, 'prereg-v2.json'), 'rb').read()).hexdigest() != PREREG_SHA:
    raise SystemExit('prereg-v2.json sha256 differs: stop')
a = sys.argv[1:]
cut = a.index('--')
J48, J192 = a[:cut], a[cut + 1:]
OUT = os.path.join(D9, 'select', 'recipes')
REP = os.path.join(D9, 'select', 'recipes.txt')
os.makedirs(OUT, exist_ok=True)
runfile.claim(REP)
lines = []


def say(s=''):
    print(s, flush=True)
    lines.append(s)


# --- the scorer's pixel set, copied ------------------------------------------------
B = 'geom/l0-1203-render/audit/build-deciding-test'
CTRL24 = 'geom/c0841/mesh0b-render/w00big'
H, W, P, E = 995, 1245, 4, 40
mask = np.load(os.path.join(B, 'ours-mask.npy'))
m24 = np.asarray(Image.open(os.path.join(CTRL24, 'mask.png')), np.float32)[:H * P, :W * P].reshape(H, P, W, P).min((1, 3)) > 0
VAL = mask & m24
M = VAL.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False


def rnorm(x):
    r = np.full(x.shape, 0.5, np.float32)
    r[M] = rankdata(x[M]) / M.sum()
    return r


SRC = {'select': os.path.join(D9, 'select', 'maps'), 'tta': os.path.join(D9, 'select', 'maps-tta'),
       'ondisk': os.path.join(D9, 'baselines', 'maps')}


def mean5(where, name):
    ps = [os.path.join(SRC[where], '%s-c%+d-rev.npy' % (name, c)) for c in (-4, -2, 0, 2, 4)]
    miss = [p for p in ps if not os.path.exists(p)]
    if miss:
        raise SystemExit('missing maps: %s' % miss)
    return np.mean([np.load(p).astype(np.float32) for p in ps], 0), ps


MIT = ('dense_native', 'klavis_dense', 'tauil_ft6k')
mem = {}
for n in MIT:
    mem[n], _ = mean5('select', n)
for n in ('soup42', 'soup43', 's43_75k'):
    mem[n], _ = mean5('ondisk', n)
tta = {}
for n in MIT:
    tta[n], _ = mean5('tta', n + '_tta')

# --- R1 by the prereg's rule --------------------------------------------------------
rows48, rows192 = {}, {}
for f in J48:
    rows48.update({r['map']: r for r in json.load(open(f))['rows']})
for f in J192:
    rows192.update({r['map']: r for r in json.load(open(f))['rows']})
ELIG = MIT + ('s43_75k', 'soup42', 'soup43')
cand = [(n, rows48['%s rev mean5' % n]['auc_sup'], rows192['%s rev mean5' % n]['hp_sp_b2']['r']) for n in ELIG]
best_auc = max(c[1] for c in cand)
tied = [c for c in cand if c[1] >= best_auc - 0.01]
r1 = max(tied, key=lambda c: c[2])
say('day1-select-recipes.py  prereg-v2 sha256 matches; score JSONs: s48 %s; s192 %s' % (J48, J192))
say('')
say('R1 candidates (rev mean5): AUCsup | hp r 192 um blur 2 vs SP')
for n, au, h in sorted(cand, key=lambda c: -c[1]):
    say('  %-14s %.4f | %+.4f%s' % (n, au, h, '   <- within 0.01 of best AUCsup' if (n, au, h) in tied else ''))
say('R1 = %s (best AUCsup %.4f; %d within 0.01, tie-break hp r 192 b2 %+.4f)' % (r1[0], best_auc, len(tied), r1[2]))
say('  mechanical, by the prereg rule; the pick between recipes is the next part\'s, not made here')
say('')

recipes = {
    'R1': (mem[r1[0]], 'mean5 of %s (identical to its member row)' % r1[0]),
    'R2': (np.mean([rnorm(mem[n]) for n in MIT], 0), 'rank-mean of %s' % '+'.join(MIT)),
    'R3': (np.mean([rnorm(tta[n]) for n in MIT], 0), 'rank-mean of %s, each with mirror TTA' % '+'.join(MIT)),
    'R4': (np.mean([rnorm(mem[n]) for n in MIT + ('soup42', 'soup43')], 0), 'rank-mean of %s' % '+'.join(MIT + ('soup42', 'soup43'))),
    'CHECKens2': (np.mean([rnorm(mem[n]) for n in ('soup42', 'soup43')], 0), "CHECK: the scorer's ENS rank-mean soup42+soup43, rebuilt here"),
}
for k, (x, what) in recipes.items():
    p = os.path.join(OUT, '%s-c+0-rev.npy' % k)
    np.save(p, x.astype(np.float32))
    say('%-10s %s; in-M mean %.4f sd %.4f; wrote %s' % (k, what, float(x[M].mean()), float(x[M].std()), p))
say('')
c = np.corrcoef(np.stack([tta[n][M] for n in MIT] + [mem[n][M] for n in MIT]))
for i, n in enumerate(MIT):
    say('  TTA vs no-TTA mean5, Pearson r over M: %-13s %.4f' % (n, c[i, i + 3]))
runfile.done(REP, '\n'.join(lines) + '\n')
