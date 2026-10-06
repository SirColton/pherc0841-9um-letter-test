"""student2-train.py - the twin-scan STUDENT v2 harness: a copy of v1's student-train.py (sha256 63289390...f565)
plus exactly the edits geom/student-v2/prereg-student-v2.json (sha256 f348e4b8..., checked first; a mismatch stops
the run) lists in 5_harness:
  (1) PREREG -> the v2 prereg and its frozen sha256. It also checks v1's prereg (f3ebf0ab...), amendment 1, both v1
      training steps.jsonl and both v1 dry-run steps.jsonl against the hashes the v2 prereg pins. v1's prereg still
      supplies every value v2 inherits (teacher, masks, distance arrays, base, w00big), read as v1's code reads it.
  (2) --out and --amendment under geom/student-v2/; outside --smoke the segments are v1's renders in
      geom/student-9um/renders/ and each one's stack, mask, spmap grid and grid.json must match the v2 prereg's sha256.
  (3) the anti-shortcut term (prereg-v2 1): L = BCE + 0.5 (1 - rho_hp) + LAMBDA x P_shuf, LAMBDA 0.5, P_shuf = sum of
      sigmoid(logit) over the batch's training px of a copy of the SAME augmented, normalised window with its 17 depth
      layers reordered (x_shuf[b, 0, j] = x[b, 0, perm_b[j]]), divided by the batch's training px count; fp32 from the
      bf16-autocast logits; target 0 in both arms. One permutation per sample from its own generator numpy
      default_rng(20261008), drawn in the producer right after that batch's clean draws, identity and full reversal
      redrawn; applied on the GPU. Per micro-batch: v1's clean forward/backward unchanged, then the permuted copy's
      forward/backward, gradients accumulating; then v1's single clip 1.0, step and schedule. No forward-order penalty.
  (4) logged only, never deciding: per step p_shuf, pen, loss_total, rho_hp_shuf (no gradient), perm_sha1 and
      clean_match_v1; VAL adds the strip read through the fixed permutation default_rng(0).permutation(17).
  (5) outputs student2-final-step012000.pth / scramble2-final-step012000.pth; LOGONLY checkpoints at 3000/6000/9000.
  CLEAN-STREAM CHECK (prereg-v2 5_harness): outside --smoke, --dry must reproduce v1's dry runs (in, mask and target
  sha1; train-student/ and train-scramble/precheck-amend1-dry), and training must match v1's training log at every
  step BEFORE that step's forward pass (in and mask sha1 against train-student/steps.jsonl, target sha1 against the
  same arm's log); the first mismatch stops the run.
  [machine] --lam is a smoke-only option (the penalty wiring test at 0); outside --smoke anything but 0.5 stops.
Everything below is v1's description and still holds, except where the lines above change it.

v1: student-train.py - the twin-scan STUDENT harness: fine-tune TAUIL ft6k on native 9.366 um renders of PHerc0841
w00 and 552896 against ScrollPrize's 2.4 um map by index, exactly as geom/student-9um/prereg-student.json
(sha256 f3ebf0ab..., checked first; a mismatch stops the run) sections 1-4 say. --scramble trains the twin.

WHAT IT READS, each checked before step 1 and the run stops on any failure:
  - the prereg (sha256), the base payload tauil_ft6k.pth (sha256; strict load, 508 tensors, 68,175,426 floats),
    the freeze masks (sha256 in the prereg) and distance arrays (sha256 in geom.json, itself hashed in the prereg)
  - each segment as day1-render-9um.py writes it, given as --seg NAME=PREFIX:
      PREFIX-stack.npy (41, H, W) uint8, stored order; PREFIX-mask.npy (H, W) bool; PREFIX-spmap-grid.npy (H, W)
      float32 = the teacher canvas by index (render px (i, j) = mean of block [20 r0 + 4i, +4) x [20 c0 + 4j, +4)
      / 255); PREFIX-grid.json (crop_cells, H, W, NL, stored_along, orient, stack_sha1)
    H = 5 x (cells - 1), so render px (i, j) belongs to cell (r0 + i // 5, c0 + j // 5) (prereg cell_to_render_px)
  - the teacher grid is RE-DERIVED from the hashed teacher tif by index and must equal the render's spmap grid
    exactly; per-cell teacher-valid must equal the freeze's reason bit 2 on every valid cell
  - w00 (full crop): its stack must equal w00big-stack.npy (sha256 in the prereg), max |diff| 0, on every px at
    least 5 px inside the window render rows [1900, 2895) cols [2500, 3745) (prereg 2_training_regions.renders.w00
    CHECK as amendment 1 item 1 narrows it); the outer 5 px frame is reported and never decides
THE 252405 EXCLUSION, two ways: (1) a Python audit hook refuses any open/listdir/scandir of a path containing
'252405' for the whole process, probed at start; (2) before step 1 the min of *-dcell3d-um over every px of
every valid window (>= 228.0 um) and of *-dflat2d-um over them (>= 2096 um, or inf = no on-sheet vertex) is
logged, over all window px and over the in-mask px (the footprint); either failing stops the run. Every drawn
window is re-checked as it is drawn and logged to footprints.csv (step, sample, segment, origin, cells, depth
centre, offset, flips, rotation, training px, min dcell3d).
TRAINING PX (both arms, prereg 4): render mask AND allowed cell AND np.roll(V_T, (416, 416)), V_T = render mask AND
teacher-valid cell. A window origin is valid when every in-mask px is in an allowed cell and >= 50% of its px are
training px. Each step: 8 windows, segment with p proportional to its valid-origin count, origin uniform in it;
depth centre c uniform on -4..+4, offset o on 0..4: stack[20+c-10 : 20+c+11] reversed, then [o, o+17).
AUGMENTATION: in-plane mirror per axis p 0.5 and rot90 x1/2/3 p 0.3 on input, target and mask together (torch
RNG); normalize_robust per patch (infer.py's tifxyz_robust path); then the members of vesuvius
create_training_transforms(patch, no_spatial=True) (the function koine_machines/data/ink_dataset.py imports) on
the input only. Depth is never flipped or permuted.
LOSS: BCE(logit, teacher) over training px of the batch + 0.5 x (1 - mean rho_hp) over samples with >= 2048 core
px; hp(x) = x - G(x m) / max(G(m), 1e-3), G sigma 48 um (truncate 4 sigma, zero outside the patch), core = m AND
G(m, 2 sigma) > 0.999. AdamW 2e-5, betas (0.9, 0.999), wd 3e-5, 500 linear warmup then cosine to 0, clip 1.0,
bf16 autocast, fp32 weights, 12,000 steps of 8, seed 20261006, no early stopping; the student is the weights after
the last step. --scramble: the target is each segment's teacher grid rolled (+416, +416) px; same seed, windows,
augmentation and pixels (steps.jsonl carries a hash of every batch's input and mask to prove it).

Machine readings, marked in the log: the HF/diffusers warmup convention (step k uses lr x k/500, so step 0 has lr
0); numpy's legacy global RNG seeded 20261006 beside torch, because the vesuvius transforms draw from both;
stub modules for SimpleITK, pandas, skimage(.data) and fft_conv_pytorch while training_transforms imports, when
the venv lacks them (each is a top-level import no member of the no_spatial pipeline calls; every stub function
raises if called; removed after the import); GaussianBlurTransform benchmark=False (direct convolution always,
the same blur without a timing race that could make the arms differ); weight decay on every parameter.

OUT (--out, under geom/student-9um/): train-run.txt (runfile banner), steps.jsonl, footprints.csv, val.jsonl,
<arm>-final-step012000.pth = {'model': fp32 state_dict, 'config': the base payload's config unchanged, 'step'}
and its sha256; ckpt-LOGONLY-step003000/6000/9000.pth kept for the log, never scored.
After step 300 the mean step time over steps 101-300 is projected to the step count; past 6 h the run stops and
writes STOP-PROJECTION.txt (the prereg's 4 h rescale is then a dated note beside the prereg, not this script).
--steps is refused outside --smoke unless --amendment names that note.

Usage (from <workdir>; GPU only through polite-run.py, a local CPU/RAM load check under 80% first):
  polite-run.py student2-train.py --out geom/student-v2/train-student [--scramble]
  smoke: ... --smoke --steps 100 [--lam 0] --seg w00=geom/day1-9um/render/w00big-republished/w00big --out geom/student-v2/harness/...
  dry (no model, no GPU): ... --dry 2
"""
import os
import sys

FORBIDDEN = '252405'
_GUARD = {'checked': 0, 'refused': 0}


def _guard(event, args):
    if event not in ('open', 'os.listdir', 'os.scandir'):
        return
    p = args[0] if args else None
    if p is None or isinstance(p, int):
        return
    try:
        s = os.fsdecode(os.fspath(p))
    except Exception:
        return
    _GUARD['checked'] += 1
    if FORBIDDEN in s:
        _GUARD['refused'] += 1
        raise PermissionError('student2-train.py: a path containing %s was refused before P5: %s' % (FORBIDDEN, s))


sys.addaudithook(_guard)

import argparse
import hashlib
import json
import math
import queue
import subprocess
import threading
import time
import types

import numpy as np
from scipy.ndimage import gaussian_filter

import runfile

KOINE = os.path.abspath('koine/ink-detection')
sys.path.insert(0, KOINE)
PREREG, PREREG_SHA = 'geom/student-v2/prereg-student-v2.json', 'f348e4b89b19af4a682c65dfb672661296f909183ead703237711e547faa33d7'
PREREG1 = 'geom/student-9um/prereg-student.json'
AMEND1 = 'geom/student-9um/prereg-student-amendment-1.txt'
V1_TRAIN = {'student': 'geom/student-9um/train-student/steps.jsonl', 'scramble': 'geom/student-9um/train-scramble/steps.jsonl'}
V1_DRY = {'student': ('geom/student-9um/train-student/precheck-amend1-dry/steps.jsonl', '45fc4065071287cc8fd86bd9ce8ae1d35f0425d8ca9daac18b083b046d8def6f'),
          'scramble': ('geom/student-9um/train-scramble/precheck-amend1-dry/steps.jsonl', 'ad7fb425833ef334fb3786b0c723f78a9503c706732d5b2531a80a1a6205249d')}
RENDERS = 'geom/student-9um/renders'
LAMBDA, PERM_SEED, VPERM_SEED = 0.5, 20261008, 0
PERM_FIRST = [4, 2, 14, 9, 1, 5, 13, 10, 8, 6, 12, 15, 3, 11, 16, 7, 0]
VPERM_PIN = [2, 10, 3, 11, 0, 4, 7, 5, 16, 12, 13, 6, 9, 14, 8, 1, 15]
FREEZE = 'geom/student-9um/freeze/geom'
BASE = 'geom/day1-9um/select/ckpt/tauil_ft6k.pth'
BASE_KEY = 'geom/day1-9um/select/ckpt/tauil_ft6k.pth (koine payload, the base)'
W00BIG = 'geom/day1-9um/render/w00big-republished/w00big'
W00BIG_KEY = 'geom/day1-9um/render/w00big-republished/w00big-stack.npy'
W00BIG_WIN = (1900, 2895, 2500, 3745)
W00BIG_EDGE = 5
TEACH = {'w00': 'geom/day1-9um/gates/s3/20260220213127-w00/ink-detection/PHerc0841-20260220213127-2.403um-0.22m-77keV-volume-'
                '20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif',
         '552896': 'geom/student-9um/freeze/teacher/PHerc0841-20260220214732-2.403um-0.22m-77keV-volume-20260319124803-'
                   '20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif'}
MESH = {'w00': 'geom/c0841/mesh-l0/w00',
        '552896': 'geom/day1-9um/gates/s3/20260220214732-auto_grown_20260220144552896/mesh/20260220214732-on-20260319124803-2.403um.tifxyz'}
FULL_CROP = {'w00': [0, 823, 0, 928], '552896': [0, 904, 0, 920]}
SEED, BATCH, STEPS, WARMUP = 20261006, 8, 12000, 500
LR, BETAS, WD, CLIP = 2e-5, (0.9, 0.999), 3e-5, 1.0
ROLL = (416, 416)
WIN, DEPTH, SLAB, NL, CEN = 128, 17, 21, 41, 20
MIN_TRAIN = WIN * WIN // 2
CUT3D, CUT2D = 228.0, 2096.0
PX_UM = 2.403 * 4
SIG48 = 48.0 / PX_UM
SIG192 = 192.0 / PX_UM
CORE_MIN, HP_W = 2048, 0.5
NULL_MAX_R = 0.10

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--seg', action='append', default=None, metavar='NAME=PREFIX')
ap.add_argument('--scramble', action='store_true')
ap.add_argument('--steps', type=int, default=None)
ap.add_argument('--warmup', type=int, default=None)
ap.add_argument('--amendment', default=None)
ap.add_argument('--drop-552896-evidence', default=None)
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--dry', type=int, default=0)
ap.add_argument('--micro', type=int, default=BATCH)
ap.add_argument('--grad-ckpt', action='store_true')
ap.add_argument('--prefetch', type=int, default=2)
ap.add_argument('--val-every', type=int, default=2000)
ap.add_argument('--log-every', type=int, default=50)
ap.add_argument('--keep', default='3000,6000,9000')
ap.add_argument('--probe', type=int, default=None)
ap.add_argument('--lam', type=float, default=LAMBDA)
a = ap.parse_args()

OUTA = os.path.abspath(a.out).replace('\\', '/')
if '/geom/student-v2/' not in OUTA or '/freeze' in OUTA or '/renders' in OUTA:
    raise SystemExit('--out must be under geom/student-v2/<part>/ and not freeze/ or renders/')
os.makedirs(a.out, exist_ok=True)
ARM = 'scramble' if a.scramble else 'student'
REP = os.path.join(a.out, 'train-run.txt')
runfile.claim(REP)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def stop(msg):
    say('STOP: ' + msg)
    runfile.done(REP, '\n'.join(lines) + '\n')
    sys.exit(2)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


T0 = time.time()
say('student2-train.py  %s  arm %s%s%s' % (time.strftime('%Y-%m-%dT%H:%M:%S%z'), ARM, '  SMOKE' if a.smoke else '',
                                           '  DRY %d batches' % a.dry if a.dry else ''))
say('  argv: %s' % ' '.join(sys.argv[1:]))

# ---- the prereg and the guard ---------------------------------------------------------------------
got = sha256(PREREG)
if got != PREREG_SHA:
    stop('prereg-student-v2.json sha256 %s != frozen %s' % (got, PREREG_SHA))
pre2 = json.load(open(PREREG, encoding='utf-8'))
say('  prereg %s sha256 %s (matches)' % (PREREG, got))


def pin(path):
    """the v2 prereg's sha256 for a path (its inputs_sha256 keys carry a note after the path)"""
    hs_ = [v for k, v in pre2['inputs_sha256'].items() if k.split(' ')[0] == path]
    if len(hs_) != 1:
        stop('the v2 prereg pins %d hashes for %s, not 1' % (len(hs_), path))
    return hs_[0]


for p_ in (PREREG1, AMEND1, V1_TRAIN['student'], V1_TRAIN['scramble']):
    h_ = sha256(p_)
    if h_ != pin(p_):
        stop('%s sha256 %s != the v2 prereg %s' % (p_, h_, pin(p_)))
    say('  v1 %s sha256 %s (matches the v2 prereg)' % (p_, h_[:16]))
for arm_, (p_, want_) in sorted(V1_DRY.items()):
    if 'steps.jsonl sha256 %s' % want_ not in pre2['5_harness']['clean_stream_check']:
        stop('the v1 dry-run hash %s is not the one the v2 prereg pins' % want_)
    h_ = sha256(p_)
    if h_ != want_:
        stop('%s sha256 %s != the v2 prereg %s' % (p_, h_, want_))
    say('  v1 dry run (%s) %s sha256 %s (matches the v2 prereg)' % (arm_, p_, h_[:16]))
# every value v2 inherits is read from v1's prereg exactly as v1's code reads it
pre = json.load(open(PREREG1, encoding='utf-8'))
try:
    open('geom/student-v2/__guard_probe_%s__' % FORBIDDEN, 'rb')
    stop('the 252405 guard did not refuse its probe')
except PermissionError:
    say('  252405 guard: audit hook live; its probe open was refused before reaching the disk')

N = STEPS if a.steps is None else a.steps
WU = WARMUP if a.warmup is None else a.warmup
if not a.smoke:
    if (a.steps not in (None, STEPS)) and not a.amendment:
        stop('--steps %s outside --smoke needs --amendment <dated note beside the prereg>' % a.steps)
    if a.amendment:
        am = os.path.abspath(a.amendment).replace('\\', '/')
        if '/geom/student-v2/' not in am or not os.path.exists(a.amendment) or str(a.steps) not in open(a.amendment, encoding='utf-8').read():
            stop('--amendment must be an existing note under geom/student-v2/ naming the step count %s' % a.steps)
        say('  steps %d per the amendment %s (sha256 %s)' % (N, a.amendment, sha256(a.amendment)))
    if a.warmup is not None and a.warmup != WARMUP:
        stop('--warmup is a smoke-test option')
    if a.lam != LAMBDA:
        stop('--lam is a smoke-test option: the v2 prereg fixes LAMBDA %.1f' % LAMBDA)
    if a.micro not in (8, 4):
        stop('--micro must be 8 or 4 (prereg 3_recipe precision)')
elif a.steps is None:
    N = 50
if a.smoke and a.warmup is None:
    WU = min(WARMUP, max(1, N // 5))
KEEP = set() if a.smoke else {int(x) for x in a.keep.split(',') if x.strip()}
PROBE = (16 if a.smoke else 0) if a.probe is None else a.probe
segs_arg = a.seg or ['w00=geom/student-9um/renders/w00', '552896=geom/student-9um/renders/552896']
SEGS = [s.split('=', 1) for s in segs_arg]
names = [n for n, _ in SEGS]
for n, p in SEGS:
    if n not in TEACH:
        stop('unknown segment %s' % n)
if not a.smoke:
    if 'w00' not in names:
        stop('w00 is required')
    if '552896' not in names:
        ev = a.drop_552896_evidence
        if not ev or not os.path.exists(ev) or 'STOP' not in open(ev, encoding='utf-8').read():
            stop('552896 is dropped only when its render run stopped on orientation: pass --drop-552896-evidence <its run.txt>')
        say('  552896 DROPPED per prereg 2_training_regions.renders.552896 (its render stopped: %s); the student trains on w00 alone' % ev)
say('  recipe: %d steps x %d windows (micro %d), warmup %d, AdamW lr %.0e betas %s wd %.0e, clip %.1f, bf16 autocast, seed %d'
    % (N, BATCH, a.micro, WU, LR, BETAS, WD, CLIP, SEED))
say('  [machine] lr schedule: step k (0-based) uses lr x k/%d for k < %d, then 0.5 lr (1 + cos(pi (k - %d) / (%d - %d))); HF/diffusers convention'
    % (WU, WU, WU, N, WU))
LAM = a.lam
say('  v2 anti-shortcut term (prereg-v2 1): loss + LAMBDA x P_shuf, LAMBDA %g%s; P_shuf = mean sigmoid(logit) over the batch\'s training px of the same'
    ' windows with their 17 depth layers permuted per sample; target 0 in both arms; no forward-order penalty'
    % (LAM, '' if LAM == LAMBDA else '  [SMOKE OVERRIDE of the prereg\'s %g]' % LAMBDA))

# ---- freeze hashes --------------------------------------------------------------------------------
gj_path = os.path.join(FREEZE, 'geom.json')
want = pre['inputs_sha256']['geom/student-9um/freeze/geom/geom.json']
if sha256(gj_path) != want:
    stop('geom.json sha256 differs from the prereg')
gjson = json.load(open(gj_path, encoding='utf-8'))
fs = dict(pre['2_training_regions']['masks']['files_sha256'])
fs = {k: v for k, v in fs.items() if k.endswith('.npy')}
fs.update(gjson['files'])
for p, h in sorted(fs.items()):
    if sha256(p) != h:
        stop('%s sha256 differs from the frozen value' % p)
say('  freeze: geom.json and %d mask/distance arrays match their frozen sha256' % len(fs))
tsha = {'w00': pre['1_teacher']['files']['w00']['sha256'], '552896': pre['1_teacher']['files']['552896']['sha256']}


def cells_to_px(arr, r0, c0, H, W):
    return arr[np.ix_(r0 + np.arange(H) // 5, c0 + np.arange(W) // 5)]


def winsum(b, k=WIN):
    S = np.zeros((b.shape[0] + 1, b.shape[1] + 1), np.int32)
    S[1:, 1:] = np.cumsum(np.cumsum(b, 0, dtype=np.int32), 1, dtype=np.int32)
    return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]


def covered_by(origins, H, W, k=WIN):
    o = np.zeros((H, W), bool)
    o[:origins.shape[0], :origins.shape[1]] = origins
    S = np.zeros((H + 1, W + 1), np.int32)
    S[1:, 1:] = np.cumsum(np.cumsum(o, 0, dtype=np.int32), 1, dtype=np.int32)
    i1, j1 = np.arange(H) + 1, np.arange(W) + 1
    i0, j0 = np.maximum(i1 - k, 0), np.maximum(j1 - k, 0)
    return (S[np.ix_(i1, j1)] - S[np.ix_(i0, j1)] - S[np.ix_(i1, j0)] + S[np.ix_(i0, j0)]) > 0


def hp_np(x, m, sig):
    return x - gaussian_filter(x * m, sig) / np.maximum(gaussian_filter(m, sig), 1e-3)


# ---- segments -------------------------------------------------------------------------------------
data = {}
for name, prefix in SEGS:
    say('')
    say('== %s  %s' % (name, prefix))
    if not a.smoke:
        if prefix.replace('\\', '/') != '%s/%s' % (RENDERS, name):
            stop('outside --smoke the segments are v1\'s renders, %s/%s (prereg-v2 5_harness)' % (RENDERS, name))
        for suf in ('-stack.npy', '-mask.npy', '-spmap-grid.npy', '-grid.json'):
            h_ = sha256(prefix + suf)
            if h_ != pin(prefix + suf):
                stop('%s%s sha256 %s != the v2 prereg %s' % (prefix, suf, h_, pin(prefix + suf)))
        say('  v1 render files %s-stack.npy, -mask.npy, -spmap-grid.npy, -grid.json: sha256 match the v2 prereg' % prefix)
    g = json.load(open(prefix + '-grid.json', encoding='utf-8'))
    syn = bool(g.get('synthetic'))
    if syn and not a.smoke:
        stop('%s is SYNTHETIC; refused outside --smoke' % prefix)
    r0, r1, c0, c1 = g['crop_cells']
    H, W = int(g['H']), int(g['W'])
    if (H, W) != (5 * (r1 - r0 - 1), 5 * (c1 - c0 - 1)) or int(g['NL']) != NL or int(g.get('pool', 4)) != 4:
        stop('grid.json geometry %s is not day1-render-9um.py\'s' % ((H, W, g['NL']),))
    if not a.smoke and list(g['crop_cells']) != FULL_CROP[name]:
        stop('crop %s is not the prereg\'s full crop %s' % (g['crop_cells'], FULL_CROP[name]))
    if not syn and g['mesh'].replace('\\', '/') != MESH[name]:
        stop('mesh %s is not the prereg\'s %s' % (g['mesh'], MESH[name]))
    orient = g.get('orient') or {}
    if name == 'w00' and g.get('stored_along') != 'plus normal':
        stop('w00 must be stored along PLUS the normal (--no-orient)')
    if name == '552896' and not syn and (not orient or orient.get('ambiguous_way1') or orient.get('ambiguous_way2')
                                         or orient.get('flip_way1') != orient.get('flip_way2')):
        stop('552896 orientation is ambiguous or the two ways disagree: the prereg drops 552896')
    say('  %s; crop cells rows [%d, %d) cols [%d, %d); render %d x %d x %d; stored along %s%s'
        % ('SYNTHETIC stack and mask' if syn else 'mesh %s' % g['mesh'], r0, r1, c0, c1, H, W, NL, g.get('stored_along'),
           ('; orient way1 median %+.3f way2 %+.3f, flipped %s' % (orient['way1_target_median'], orient['way2_dot_median'], g.get('flipped'))) if orient else ''))
    stack = np.load(prefix + '-stack.npy')
    mask = np.load(prefix + '-mask.npy')
    grid = np.load(prefix + '-spmap-grid.npy')
    if stack.shape != (NL, H, W) or stack.dtype != np.uint8 or mask.shape != (H, W) or mask.dtype != bool or grid.shape != (H, W) or grid.dtype != np.float32:
        stop('render arrays have the wrong shape or dtype')
    s1 = hashlib.sha1(stack.tobytes()).hexdigest()[:12]
    if s1 != g['stack_sha1']:
        stop('stack sha1 %s != grid.json %s' % (s1, g['stack_sha1']))
    say('  stack sha1 %s (= grid.json); mask %d px (%.2f cm2)' % (s1, mask.sum(), mask.sum() * (PX_UM * 1e-4) ** 2))
    tif = TEACH[name]
    th = sha256(tif)
    if th != tsha[name]:
        stop('teacher tif sha256 %s != prereg %s' % (th, tsha[name]))
    import tifffile
    cv = tifffile.imread(tif)
    cv = cv.reshape(cv.shape[-2:]) if cv.ndim > 2 else cv
    blk = cv[20 * r0:20 * r0 + H * 4, 20 * c0:20 * c0 + W * 4]
    tg = blk.astype(np.float32).reshape(H, 4, W, 4).mean((1, 3)) / 255.0
    del cv, blk
    if not np.array_equal(tg, grid):
        stop('the render\'s spmap grid is not the teacher by index (max |diff| %g)' % float(np.abs(tg - grid).max()))
    del tg
    say('  teacher tif sha256 %s (matches the prereg); the render\'s spmap grid equals it by index, max |diff| 0' % th[:16])
    fz = {t: np.load(os.path.join(FREEZE, '%s-%s.npy' % (name, t))) for t in ('allowed', 'reason', 'dcell3d-um', 'dflat2d-um')}
    allowed_px = cells_to_px(fz['allowed'], r0, c0, H, W)
    tv_cell = grid.reshape(H // 5, 5, W // 5, 5).max((1, 3)) > 0
    rs = fz['reason'][r0:r0 + H // 5, c0:c0 + W // 5]
    okc = (rs & 1) == 0
    mism = int((okc & (((rs & 2) > 0) != ~tv_cell)).sum())
    if mism:
        stop('%d valid cells disagree with the freeze on teacher-valid' % mism)
    say('  teacher-valid per cell equals the freeze\'s reason bit 2 on all %d valid cells of the crop (%d teacher-empty)'
        % (okc.sum(), int((okc & ~tv_cell).sum())))
    tv_px = np.repeat(np.repeat(tv_cell, 5, 0), 5, 1)
    VT = mask & tv_px
    R = np.roll(VT, ROLL, (0, 1))
    T = mask & allowed_px & R
    target = np.roll(grid, ROLL, (0, 1)) if a.scramble else grid
    say('  px: mask %d, allowed %d, V_T %d, roll(V_T) %d -> TRAINING PX %d (%.2f cm2); in mask & allowed but lost to the roll %d'
        % (mask.sum(), (mask & allowed_px).sum(), VT.sum(), R.sum(), T.sum(), T.sum() * (PX_UM * 1e-4) ** 2, (mask & allowed_px & ~R).sum()))
    say('  target: %s' % ('teacher grid ROLLED (+%d, +%d) px with wrap (scramble twin)' % ROLL if a.scramble else 'teacher grid by index'))
    # twin null check (prereg 4): hp r at 192 um between the teacher grid and its rolled grid over the training px
    mf = T.astype(np.float32)
    ha, hb = hp_np(grid, mf, SIG192), hp_np(np.roll(grid, ROLL, (0, 1)), mf, SIG192)
    r_tr = float(np.corrcoef(ha[T], hb[T])[0, 1])
    core192 = T & (gaussian_filter(mf, 2 * SIG192) > 0.999)
    r_core = float(np.corrcoef(ha[core192], hb[core192])[0, 1]) if core192.sum() > 100 else float('nan')
    null_ok = abs(r_tr) <= NULL_MAX_R
    say('  twin null check: hp r 192 um (sigma %.3f px) teacher vs rolled teacher over the training px %+.4f (over its core, %d px, %+.4f) -> %s'
        % (SIG192, r_tr, core192.sum(), r_core, 'a valid null (|r| <= 0.10)' if null_ok else 'NOT A NULL (|r| > 0.10): the verdict cannot be GO (INCONCLUSIVE)'))
    del ha, hb, mf, core192
    # valid window origins
    bad = mask & ~allowed_px
    ok_o = (winsum(bad) == 0) & (winsum(T) >= MIN_TRAIN)
    Ho, Wo = ok_o.shape
    origins = np.flatnonzero(ok_o).astype(np.int64)
    if len(origins) == 0:
        stop('no valid window origin in %s' % name)
    cov = covered_by(ok_o, H, W)
    dc = cells_to_px(fz['dcell3d-um'], r0, c0, H, W)
    df = cells_to_px(fz['dflat2d-um'], r0, c0, H, W)
    d3_all, d3_in = float(dc[cov].min()), float(dc[cov & mask].min())
    d2_all, d2_in = float(df[cov].min()), float(df[cov & mask].min())
    say('  valid window origins %d of %d (%.1f%%); px covered by some valid window %d (%.2f cm2), %d of them in the mask'
        % (len(origins), Ho * Wo, 100 * len(origins) / (Ho * Wo), cov.sum(), cov.sum() * (PX_UM * 1e-4) ** 2, (cov & mask).sum()))
    say('  ASSERT 3D: min dcell3d over every px of every valid window %.1f um (in-mask footprint %.1f) >= %.1f: %s'
        % (d3_all, d3_in, CUT3D, 'PASS' if min(d3_all, d3_in) >= CUT3D else 'FAIL'))
    say('  ASSERT 2D: min dflat2d over them %s (in-mask %s) >= %.0f or no on-sheet vertex: %s'
        % ('inf (no on-sheet vertex)' if not np.isfinite(d2_all) else '%.1f um' % d2_all,
           'inf' if not np.isfinite(d2_in) else '%.1f um' % d2_in, CUT2D, 'PASS' if min(d2_all, d2_in) >= CUT2D else 'FAIL'))
    if min(d3_all, d3_in) < CUT3D or min(d2_all, d2_in) < CUT2D:
        stop('a valid window reaches inside a 252405 cut')
    del cov, df
    if name == 'w00' and list(g['crop_cells']) == FULL_CROP['w00']:
        if sha256(W00BIG + '-stack.npy') != pre['inputs_sha256'][W00BIG_KEY]:
            stop('w00big-stack.npy sha256 differs from the prereg')
        rb, re_, cb, ce = W00BIG_WIN
        ref, refm = np.load(W00BIG + '-stack.npy'), np.load(W00BIG + '-mask.npy')
        both = mask[rb:re_, cb:ce] & refm
        ad = np.abs(stack[:, rb:re_, cb:ce].astype(np.int16) - ref.astype(np.int16))
        # Amendment 1 item 1: only px at least W00BIG_EDGE px inside the window decide (day 1 rendered w00big from a
        # crop, so its frame has one-sided normals). [machine] Every interior px, masks or not: a superset of the
        # prereg's both-masks set, so a pass holds under either reading.
        E = W00BIG_EDGE
        inner = np.zeros(both.shape, bool)
        inner[E:-E, E:-E] = True
        frame = both & ~inner
        dif = int(ad[:, inner].max())
        say('  CHECK w00 vs w00big-stack.npy (amendment 1 item 1) on the interior, rows [%d, %d) cols [%d, %d): %d px, %d of them where both masks are set: max |diff| %d -> %s'
            % (rb + E, re_ - E, cb + E, ce - E, inner.sum(), (both & inner).sum(), dif, 'PASS' if dif == 0 else 'FAIL'))
        say('    frame, the outer %d px of rows [%d, %d) cols [%d, %d) (reported, never decides): %d px where both masks are set, %d voxels differ, max |diff| %d'
            % (E, rb, re_, cb, ce, frame.sum(), int((ad[:, frame] > 0).sum()), int(ad[:, frame].max()) if frame.any() else 0))
        if dif:
            stop('the w00 render does not reproduce w00big on the interior (amendment 1 item 1)')
        del ref, refm, ad
    elif name == 'w00':
        say('  CHECK w00 vs w00big: not run, this w00 crop is %s (smoke)' % (g['crop_cells'],))
    val = None
    if name == 'w00':
        vs = cells_to_px(np.load(os.path.join(FREEZE, 'w00-valstrip.npy')), r0, c0, H, W) & mask
        if vs.any() and not (vs & T).any():
            ys, xs = np.nonzero(vs)
            y_a, y_b, x_a, x_b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

            def starts(lo, hi, lim):
                s = list(range(lo, max(lo, hi - WIN) + 1, WIN))
                if s[-1] + WIN < hi:
                    s.append(hi - WIN)
                return [min(max(0, v), lim - WIN) for v in s]
            vw = [(y, x) for y in starts(y_a, y_b, H) for x in starts(x_a, x_b, W)]
            val = {'px_own': vs & (R if a.scramble else VT), 'px_teacher': vs & VT, 'windows': vw}
            say('  validation strip: %d px in the crop (rows [%d, %d) cols [%d, %d)), %d windows; never a training px' % (vs.sum(), y_a, y_b, x_a, x_b, len(vw)))
        elif (vs & T).any():
            stop('the validation strip overlaps the training px')
    data[name] = {'stack': stack, 'mask': mask, 'grid': grid, 'target': target, 'T': T, 'allowed': allowed_px, 'dc': dc,
                  'origins': origins, 'Wo': Wo, 'crop': (r0, r1, c0, c1), 'H': H, 'W': W, 'val': val, 'null_r': r_tr, 'null_ok': null_ok}

names = [n for n, _ in SEGS]
counts = np.array([len(data[n]['origins']) for n in names], np.float64)
probs = counts / counts.sum()
say('')
say('  segment draw p (prop. to valid origins): %s' % ', '.join('%s %.4f (%d)' % (n, p, c) for n, p, c in zip(names, probs, counts.astype(int))))
say('  window area per step %.2f cm2; %d steps -> %.0f cm2 over %.2f cm2 of training px (%.1f passes)'
    % (BATCH * (WIN * PX_UM * 1e-4) ** 2, N, N * BATCH * (WIN * PX_UM * 1e-4) ** 2,
       sum(data[n]['T'].sum() for n in names) * (PX_UM * 1e-4) ** 2,
       N * BATCH * WIN * WIN / max(1, sum(data[n]['T'].sum() for n in names))))
twin_null = all(data[n]['null_ok'] for n in names)
say('  twin a valid null on every segment: %s' % twin_null)

# ---- augmentation ---------------------------------------------------------------------------------
import torch

src = open(os.path.join(KOINE, 'koine_machines', 'data', 'ink_dataset.py'), encoding='utf-8').read()
IMP = 'from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms'
if IMP not in src:
    stop('ink_dataset.py no longer imports create_training_transforms from vesuvius')
# Top-level imports in the transforms chain that this venv lacks and that no member of the no_spatial pipeline
# calls: spatial.py's SimpleITK (its __main__ block only) and pandas (never used); gaussian_blur.py's
# skimage.data.camera (never used) and fft_conv_pytorch.fft_conv (only on the benchmark=True timing race, turned
# off below so the direct convolution always runs). Each stub attribute raises if called; the stubs leave
# sys.modules after this import, so nothing later sees a fake module.
STUBBABLE = {'SimpleITK': (), 'pandas': (), 'skimage': (), 'skimage.data': ('camera',), 'fft_conv_pytorch': ('fft_conv',)}


def _dead(*_a, **_k):
    raise RuntimeError('a stubbed module function was called: the stub reading is wrong, stop')


stubs = []
for _ in range(len(STUBBABLE) + 1):
    try:
        from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
        break
    except ModuleNotFoundError as e:
        if e.name in STUBBABLE and e.name not in sys.modules:
            mod = types.ModuleType(e.name)
            for at in STUBBABLE[e.name]:
                setattr(mod, at, _dead)
            sys.modules[e.name] = mod
            stubs.append(e.name)
            continue
        raise
for s_ in stubs:
    del sys.modules[s_]
stub = bool(stubs)
from vesuvius.image_proc.intensity.normalization import normalize_robust
AUG = create_training_transforms((DEPTH, WIN, WIN), no_spatial=True)


def walk(t, p=None):
    kids = list(getattr(t, 'transforms', []) or []) + list(getattr(t, 'list_of_transforms', []) or [])
    if hasattr(t, 'transform'):
        yield from walk(t.transform, getattr(t, 'apply_probability', None))
        return
    yield t, type(t).__name__, p
    for k in kids:
        yield from walk(k)


members = list(walk(AUG))
SPATIAL = {'SpatialTransform', 'MirrorTransform', 'Rot90Transform', 'TransposeAxesTransform'}
if any(n in SPATIAL for _, n, _ in members):
    stop('a spatial member is in the no_spatial pipeline')
nblur = 0
for t_, n_, _ in members:
    if n_ == 'GaussianBlurTransform':
        t_.benchmark = False
        nblur += 1
say('  intensity augmentation: vesuvius create_training_transforms((17, 128, 128), no_spatial=True), the function ink_dataset.py imports; members:')
say('    ' + ', '.join('%s%s' % (n, '' if p is None else ' p%.2g' % p) for _, n, p in members if n not in ('ComposeTransforms', 'OneOfTransform')))
say('  [machine] GaussianBlurTransform benchmark=False on its %d member(s): the direct convolution every time instead of a per-call timing race between it and'
    ' an FFT convolution of the same kernel; same blur, and the two arms cannot diverge on a timing' % nblur)
if stub:
    say('  [machine] absent from the venv and stubbed only while training_transforms imported, then removed: %s (spatial.py imports SimpleITK for its '
        '__main__ block and pandas unused; gaussian_blur.py imports skimage.data.camera unused and fft_conv only for the timing race, now off); '
        'every stub function raises if called; no spatial member is in this pipeline (asserted)' % ', '.join(stubs))

# ---- seeds and the producer -----------------------------------------------------------------------
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
say('  seeds: numpy default_rng(%d) draws segment, origin, depth centre and offset; torch.manual_seed(%d) the flips, rotation and torch-side intensity draws; [machine] np.random.seed(%d) the vesuvius transforms\' numpy-side draws' % (SEED, SEED, SEED))
# v2: separate numpy Generators, so the global numpy/torch RNGs and default_rng(SEED) see exactly v1's draws
prng_perm = np.random.default_rng(PERM_SEED)
IDP, REVP = np.arange(DEPTH), np.arange(DEPTH)[::-1]
NREDRAW = [0]
if np.random.default_rng(PERM_SEED).permutation(DEPTH).tolist() != PERM_FIRST:
    stop('default_rng(%d).permutation(17) is not the prereg\'s pinned first permutation' % PERM_SEED)
VPERM = np.random.default_rng(VPERM_SEED).permutation(DEPTH)
if VPERM.tolist() != VPERM_PIN:
    stop('default_rng(%d).permutation(17) is not the prereg\'s pinned VAL permutation' % VPERM_SEED)
say('  v2 permutations: numpy default_rng(%d), its own generator, 8 per step in sample order right after the clean draws, identity and full reversal redrawn;'
    ' first draw %s (= the prereg\'s pin). VAL strip permutation default_rng(%d).permutation(17) = %s (= the pin)'
    % (PERM_SEED, PERM_FIRST, VPERM_SEED, VPERM.tolist()))


def draw_perm():
    while True:
        p_ = prng_perm.permutation(DEPTH)
        if not (np.array_equal(p_, IDP) or np.array_equal(p_, REVP)):
            return p_
        NREDRAW[0] += 1


def window(seg, y0, x0, c, o):
    d = data[seg]
    raw = d['stack'][:, y0:y0 + WIN, x0:x0 + WIN]
    img = np.ascontiguousarray(raw[CEN + c - 10:CEN + c + 11][::-1][o:o + DEPTH]).astype(np.float32)
    return img, d['target'][y0:y0 + WIN, x0:x0 + WIN].copy(), d['T'][y0:y0 + WIN, x0:x0 + WIN].copy()


def check_window(seg, y0, x0):
    d = data[seg]
    m, al, T = d['mask'][y0:y0 + WIN, x0:x0 + WIN], d['allowed'][y0:y0 + WIN, x0:x0 + WIN], d['T'][y0:y0 + WIN, x0:x0 + WIN]
    md = float(d['dc'][y0:y0 + WIN, x0:x0 + WIN][m].min()) if m.any() else float('inf')
    if (m & ~al).any() or T.sum() < MIN_TRAIN or md < CUT3D:
        raise RuntimeError('window %s (%d, %d) breaks the training-px rule' % (seg, y0, x0))
    return int(T.sum()), md


def make_batch(step):
    xs, ts, ms, rows = [], [], [], []
    for k in range(BATCH):
        si = int(rng.choice(len(names), p=probs))
        seg = names[si]
        d = data[seg]
        idx = int(d['origins'][int(rng.integers(len(d['origins'])))])
        y0, x0 = divmod(idx, d['Wo'])
        c, o = int(rng.integers(-4, 5)), int(rng.integers(0, 5))
        ntr, md = check_window(seg, y0, x0)
        img, tgt, m = window(seg, y0, x0, c, o)
        fy, fx = bool(torch.rand(1).item() < 0.5), bool(torch.rand(1).item() < 0.5)
        rot = int(torch.randint(1, 4, (1,)).item()) if torch.rand(1).item() < 0.3 else 0
        if fy:
            img, tgt, m = img[:, ::-1], tgt[::-1], m[::-1]
        if fx:
            img, tgt, m = img[:, :, ::-1], tgt[:, ::-1], m[:, ::-1]
        if rot:
            img, tgt, m = np.rot90(img, rot, (1, 2)), np.rot90(tgt, rot), np.rot90(m, rot)
        img = normalize_robust(np.ascontiguousarray(img, np.float32))
        out = AUG(image=torch.from_numpy(np.ascontiguousarray(img))[None])['image']
        if tuple(out.shape) != (1, DEPTH, WIN, WIN) or not torch.isfinite(out).all():
            raise RuntimeError('augmentation changed the shape or made a non-finite value')
        xs.append(out.float())
        ts.append(torch.from_numpy(np.ascontiguousarray(tgt, np.float32))[None])
        ms.append(torch.from_numpy(np.ascontiguousarray(m, np.float32))[None])
        r0, _, c0, _ = d['crop']
        rows.append((step, k, seg, y0, x0, r0 + y0 // 5, r0 + (y0 + WIN - 1) // 5 + 1, c0 + x0 // 5, c0 + (x0 + WIN - 1) // 5 + 1,
                     c, o, int(fy), int(fx), rot, ntr, round(md, 1)))
    x, t, m = torch.stack(xs), torch.stack(ts), torch.stack(ms)
    hx = hashlib.sha1(x.numpy().tobytes()).hexdigest()[:12]
    hm = hashlib.sha1(m.numpy().tobytes()).hexdigest()[:12]
    ht = hashlib.sha1(t.numpy().tobytes()).hexdigest()[:12]
    perms = np.stack([draw_perm() for _ in range(BATCH)])
    hpm = hashlib.sha1(perms.astype(np.int8).tobytes()).hexdigest()[:12]
    return x, t, m, rows, (hx, hm, ht, hpm), perms


fp = open(os.path.join(a.out, 'footprints.csv'), 'w', encoding='utf-8')
fp.write('step,sample,segment,y0,x0,cell_r0,cell_r1,cell_c0,cell_c1,depth_centre,offset,flip_y,flip_x,rot90,train_px,min_dcell3d_um\n')
sj = open(os.path.join(a.out, 'steps.jsonl'), 'w', encoding='utf-8')


def log_rows(rows):
    for r in rows:
        fp.write(','.join(str(v) for v in r) + '\n')


q = queue.Queue(maxsize=max(1, a.prefetch))
NB = a.dry if a.dry else N


def load_ref(path, keys):
    out = {}
    for ln in open(path, encoding='utf-8'):
        r_ = json.loads(ln)
        out[int(r_['step'])] = tuple(r_[k] for k in keys)
    return out


# clean-stream check (prereg-v2 5_harness): only on the real renders, outside --smoke
REF = None
if not a.smoke:
    if a.dry:
        REF = load_ref(V1_DRY[ARM][0], ('in_sha1', 'mask_sha1', 'target_sha1'))
        say('  clean-stream check (a): each dry batch must equal v1\'s %s dry run (%d batches logged there)' % (ARM, len(REF)))
    else:
        rin = load_ref(V1_TRAIN['student'], ('in_sha1', 'mask_sha1'))
        rtg = load_ref(V1_TRAIN[ARM], ('target_sha1',))
        REF = {k: rin[k] + rtg[k] for k in rin if k in rtg}
        say('  clean-stream check (b): every step\'s in and mask sha1 must equal v1\'s %s and its target sha1 v1\'s %s before its forward pass (%d steps logged there)'
            % (V1_TRAIN['student'], V1_TRAIN[ARM], len(REF)))
else:
    say('  clean-stream check: not run (smoke; v1\'s logs are of the real renders)')


def producer():
    try:
        for s in range(1, NB + 1):
            t1 = time.perf_counter()
            b = make_batch(s)
            q.put((b, time.perf_counter() - t1))
    except BaseException as e:
        q.put(e)


if a.dry:
    say('')
    say('DRY: %d batches through the producer, no model' % a.dry)
    th = threading.Thread(target=producer, daemon=True)
    th.start()
    tb = []
    for s in range(1, a.dry + 1):
        item = q.get()
        if isinstance(item, BaseException):
            stop('producer: %r' % item)
        (x, t, m, rows, hs, perms), dt = item
        tb.append(dt)
        log_rows(rows)
        cm = None if REF is None or s not in REF else (hs[0], hs[1], hs[2]) == REF[s]
        sj.write(json.dumps({'step': s, 'in_sha1': hs[0], 'mask_sha1': hs[1], 'target_sha1': hs[2], 'perm_sha1': hs[3], 'clean_match_v1': cm,
                             'data_s': round(dt, 4), 'segments': [r[2] for r in rows], 'train_px': int(m.sum())}) + '\n')
        say('  batch %d: %s  in %s mask %s target %s  perm %s (first %s)  v1 dry match %s  %.3f s'
            % (s, ' '.join('%s(%d,%d)c%+d o%d' % (r[2], r[3], r[4], r[9], r[10]) for r in rows[:3]) + ' ...', hs[0], hs[1], hs[2], hs[3],
               perms[0].tolist(), 'not checked' if cm is None else cm, dt))
        if cm is False:
            sj.close()
            stop('clean-stream check (a): dry batch %d (in %s mask %s target %s) != v1\'s dry run %s' % (s, hs[0], hs[1], hs[2], REF[s]))
    fp.close()
    sj.close()
    say('DRY done: producer %.3f s per batch of %d (mean of %d); %d footprints logged; permutation redraws %d; guard checked %d opens, refused %d (its probe)'
        % (float(np.mean(tb)), BATCH, len(tb), a.dry * BATCH, NREDRAW[0], _GUARD['checked'], _GUARD['refused']))
    if REF is not None:
        say('  clean-stream check (a): %d of %d dry batches compared with v1\'s and all equal' % (sum(1 for s in range(1, a.dry + 1) if s in REF), a.dry))
    runfile.done(REP, '\n'.join(lines) + '\n')
    sys.exit(0)

# ---- model ----------------------------------------------------------------------------------------
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from koine_machines.inference.infer import normalize_training_config_for_inference, extract_state_dict_entry_from_payload
from koine_machines.models.make_model import make_model

bh = sha256(BASE)
if bh != pre['inputs_sha256'][BASE_KEY]:
    stop('base payload sha256 %s differs from the prereg' % bh)
basep = torch.load(BASE, map_location='cpu', weights_only=True)
base_cfg = basep['config']
model = make_model(normalize_training_config_for_inference(base_cfg))
inc = model.load_state_dict(basep['model'], strict=True)
nfl = sum(t.numel() for t in basep['model'].values() if t.is_floating_point())
if inc.missing_keys or inc.unexpected_keys or len(basep['model']) != 508 or nfl != 68175426:
    stop('base did not load as frozen: missing %d unexpected %d tensors %d floats %d' % (len(inc.missing_keys), len(inc.unexpected_keys), len(basep['model']), nfl))
say('')
say('  base %s sha256 %s (matches); strict load: missing 0, unexpected 0, %d tensors, %d floats; step %s' % (BASE, bh[:16], len(basep['model']), nfl, basep['step']))
dev = torch.device('cuda')
torch.backends.cudnn.benchmark = True
model = model.to(dev).float()
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k / WU) if k < WU else 0.5 * (1 + math.cos(math.pi * (k - WU) / max(1, N - WU))))
say('  GPU %s; cudnn.benchmark True (nondeterministic kernels allowed, prereg 3_recipe seed); grad checkpointing %s; micro-batch %d'
    % (torch.cuda.get_device_name(0), 'ON' if a.grad_ckpt else 'off', a.micro))


def gk(sig):
    r = int(4 * sig + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float64)
    k = torch.exp(-0.5 * (x / sig) ** 2)
    return (k / k.sum()).float().to(dev)


K48, K96 = gk(SIG48), gk(2 * SIG48)


def blur(x, k):
    r = (k.numel() - 1) // 2
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))


def core_of(m):
    with torch.no_grad():
        core = (m > 0) & (blur(m, K96) > 0.999)
    return core, core.flatten(1).sum(1)


def loss_parts(logits, t, m, core, ncore):
    """BCE sum over training px, and the sum of rho_hp over samples with >= CORE_MIN core px."""
    logits = logits.float()
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction='none') * m).sum()
    p = torch.sigmoid(logits)
    den = blur(m, K48).clamp_min(1e-3)
    hpp = p - blur(p * m, K48) / den
    hpt = t - blur(t * m, K48) / den
    rho, nq = logits.new_zeros(()), 0
    for b in range(logits.shape[0]):
        if int(ncore[b]) >= CORE_MIN:
            c = core[b]
            u, v = hpp[b][c], hpt[b][c]
            u, v = u - u.mean(), v - v.mean()
            rho = rho + (u * v).sum() / torch.sqrt((u * u).sum() * (v * v).sum()).clamp_min(1e-12)
            nq += 1
    return bce, rho, nq


def forward(x):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if a.grad_ckpt:
            return checkpoint(lambda inp: model(inp)['ink'], x, use_reentrant=False)
        return model(x)['ink']


def evaluate(wins, seg, own=True, px=None, perm=None):
    """loss of fixed windows (depth centre 0, offset 2, no augmentation): BCE over px, mean rho over qualifying windows,
    mean p over px. perm (v2, logged only): the normalised windows' 17 layers reordered by it before the forward."""
    d = data[seg]
    bs, ns, rs, nq, ps = 0.0, 0.0, 0.0, 0, 0.0
    pt = None if perm is None else torch.as_tensor(np.asarray(perm), dtype=torch.long, device=dev)
    with torch.no_grad():
        for i in range(0, len(wins), BATCH):
            xs, ts, ms = [], [], []
            for (y0, x0) in wins[i:i + BATCH]:
                img, _, _ = window(seg, y0, x0, 0, 2)
                xs.append(torch.from_numpy(normalize_robust(img))[None])
                tsrc = d['target'] if own else d['grid']
                ts.append(torch.from_numpy(np.ascontiguousarray(tsrc[y0:y0 + WIN, x0:x0 + WIN], np.float32))[None])
                pm = (d['T'] if px is None else px)[y0:y0 + WIN, x0:x0 + WIN]
                ms.append(torch.from_numpy(pm.astype(np.float32))[None])
            x, t, m = torch.stack(xs).to(dev), torch.stack(ts).to(dev), torch.stack(ms).to(dev)
            if pt is not None:
                x = x[:, :, pt]
            core, ncore = core_of(m)
            lg = forward(x)
            b, r, n = loss_parts(lg, t, m, core, ncore)
            bs += float(b); ns += float(m.sum()); rs += float(r); nq += n
            ps += float((torch.sigmoid(lg.float()) * m).sum())
    bce = bs / max(ns, 1)
    rho = rs / nq if nq else float('nan')
    return {'bce': bce, 'rho_hp': rho, 'n_rho': nq, 'loss': bce + (HP_W * (1 - rho) if nq else 0.0), 'px': int(ns), 'p_mean': ps / max(ns, 1)}


vj = open(os.path.join(a.out, 'val.jsonl'), 'w', encoding='utf-8')


def run_val(step):
    for n in names:
        v = data[n]['val']
        if v is None:
            continue
        rec = {'step': step, 'segment': n}
        rec['own'] = evaluate(v['windows'], n, own=True, px=v['px_own'])
        if a.scramble:
            rec['teacher'] = evaluate(v['windows'], n, own=False, px=v['px_teacher'])
        rec['own_shuf'] = evaluate(v['windows'], n, own=True, px=v['px_own'], perm=VPERM)
        vj.write(json.dumps(rec) + '\n')
        vj.flush()
        say('  VAL step %d (strip, logged only, chooses nothing): own target loss %.4f (bce %.4f, rho_hp %.4f on %d windows, mean p %.4f)%s'
            % (step, rec['own']['loss'], rec['own']['bce'], rec['own']['rho_hp'], rec['own']['n_rho'], rec['own']['p_mean'],
               ('; vs the teacher bce %.4f rho_hp %.4f' % (rec['teacher']['bce'], rec['teacher']['rho_hp'])) if a.scramble else ''))
        say('    VAL step %d through the fixed depth permutation (logged only): bce %.4f, mean p %.4f, rho_hp %.4f on %d windows'
            % (step, rec['own_shuf']['bce'], rec['own_shuf']['p_mean'], rec['own_shuf']['rho_hp'], rec['own_shuf']['n_rho']))


probe = []
if PROBE:
    prng = np.random.default_rng(SEED + 1)
    for _ in range(PROBE):
        si = int(prng.choice(len(names), p=probs))
        dd = data[names[si]]
        probe.append((names[si], divmod(int(dd['origins'][int(prng.integers(len(dd['origins'])))]), dd['Wo'])))


def run_probe():
    out = []
    for seg in names:
        w = [yx for s, yx in probe if s == seg]
        if w:
            out.append((seg, evaluate(w, seg)))
    tot_px = sum(r['px'] for _, r in out)
    bce = sum(r['bce'] * r['px'] for _, r in out) / max(tot_px, 1)
    nq = sum(r['n_rho'] for _, r in out)
    rho = sum(r['rho_hp'] * r['n_rho'] for _, r in out if r['n_rho']) / nq if nq else float('nan')
    return {'bce': bce, 'rho_hp': rho, 'n_rho': nq, 'loss': bce + (HP_W * (1 - rho) if nq else 0.0)}


def smi():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return 'nvidia-smi failed: %r' % e


say('  nvidia-smi before training (used MiB, total MiB, util %%): %s' % smi())
if probe:
    pb = run_probe()
    say('  PROBE before (%d fixed held-in windows, own target, no augmentation): loss %.4f bce %.4f rho_hp %.4f (%d)'
        % (len(probe), pb['loss'], pb['bce'], pb['rho_hp'], pb['n_rho']))
run_val(0)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# ---- train ----------------------------------------------------------------------------------------
say('')
say('TRAIN %s: %d steps' % (ARM, N))
th = threading.Thread(target=producer, daemon=True)
th.start()
t_end, times, hp_zero, losses, smi_log = time.perf_counter(), [], 0, [], {}
pshufs, rhoshufs, n_match = [], [], 0
for s in range(1, N + 1):
    tw = time.perf_counter()
    item = q.get()
    if isinstance(item, BaseException):
        stop('producer at step %d: %r' % (s, item))
    (x, t, m, rows, hs, perms), dprod = item
    wait = time.perf_counter() - tw
    log_rows(rows)
    # clean-stream check (b), before this step's forward pass
    cm = None if REF is None or s not in REF else (hs[0], hs[1], hs[2]) == REF[s]
    if cm is False:
        sj.write(json.dumps({'step': s, 'in_sha1': hs[0], 'mask_sha1': hs[1], 'target_sha1': hs[2], 'perm_sha1': hs[3], 'clean_match_v1': False,
                             'v1': list(REF[s])}) + '\n')
        sj.close()
        fp.close()
        stop('clean-stream check (b): step %d batch (in %s mask %s target %s) != v1\'s training log %s; no forward pass ran for it' % (s, hs[0], hs[1], hs[2], REF[s]))
    n_match += cm is True
    x, t, m = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
    # v2: the same windows with each sample's 17 depth layers reordered, x_shuf[b, 0, j] = x[b, 0, perm_b[j]]
    pidx = torch.from_numpy(perms).to(dev)
    x_shuf = torch.gather(x, 2, pidx.view(BATCH, 1, DEPTH, 1, 1).expand(BATCH, 1, DEPTH, WIN, WIN))
    if s == 1 and not all(torch.equal(x_shuf[b_, 0, j_], x[b_, 0, int(perms[b_, j_])]) for b_ in range(BATCH) for j_ in range(DEPTH)):
        stop('the permuted copy is not x[b, 0, perm_b[j]] at step 1')
    core, ncore = core_of(m)
    npx = float(m.sum())
    nq_tot = int((ncore >= CORE_MIN).sum())
    lr_now = opt.param_groups[0]['lr']
    opt.zero_grad(set_to_none=True)
    bce_tot, rho_tot, ps_tot, rhos_tot = 0.0, 0.0, 0.0, 0.0
    for i in range(0, BATCH, a.micro):
        sl = slice(i, i + a.micro)
        logits = forward(x[sl])
        b, r, n = loss_parts(logits, t[sl], m[sl], core[sl], ncore[sl])
        part = b / npx - (HP_W * r / nq_tot if nq_tot else 0.0)
        part.backward()
        bce_tot += float(b.detach())
        rho_tot += float(r.detach()) if n else 0.0
        # v2 anti-shortcut term (prereg-v2 1): LAMBDA x sum(sigmoid(logit_shuf) x m) / npx, gradients accumulating
        ls = forward(x_shuf[sl])
        psum = (torch.sigmoid(ls.float()) * m[sl]).sum()
        (LAM * psum / npx).backward()
        ps_tot += float(psum.detach())
        with torch.no_grad():
            _, rs_, ns_ = loss_parts(ls.detach(), t[sl], m[sl], core[sl], ncore[sl])
        rhos_tot += float(rs_) if ns_ else 0.0
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
    opt.step()
    sched.step()
    torch.cuda.synchronize()
    now = time.perf_counter()
    dt, t_end = now - t_end, now
    times.append(dt)
    bce = bce_tot / npx
    rho = rho_tot / nq_tot if nq_tot else float('nan')
    loss = bce + (HP_W * (1 - rho) if nq_tot else 0.0)
    hp_zero += nq_tot == 0
    losses.append(loss)
    p_shuf = ps_tot / npx
    pen = LAM * p_shuf
    loss_total = loss + pen
    rho_s = rhos_tot / nq_tot if nq_tot else float('nan')
    pshufs.append(p_shuf)
    rhoshufs.append(rho_s)
    if not math.isfinite(loss) or not math.isfinite(loss_total):
        stop('non-finite loss at step %d' % s)
    rec = {'step': s, 'loss': round(loss, 6), 'bce': round(bce, 6), 'rho_hp': None if not nq_tot else round(rho, 6), 'n_rho': nq_tot,
           'lr': lr_now, 'grad_norm': round(gn, 5), 'step_s': round(dt, 4), 'wait_s': round(wait, 4), 'producer_s': round(dprod, 4),
           'train_px': int(npx), 'in_sha1': hs[0], 'mask_sha1': hs[1], 'target_sha1': hs[2],
           'vram_peak_alloc_mb': round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           'p_shuf': round(p_shuf, 6), 'pen': round(pen, 6), 'loss_total': round(loss_total, 6),
           'rho_hp_shuf': None if not nq_tot else round(rho_s, 6), 'perm_sha1': hs[3], 'clean_match_v1': cm}
    if s in (10, 25) or s == N or s % 1000 == 0:
        smi_log[s] = smi()
        rec['nvidia_smi'] = smi_log[s]
    sj.write(json.dumps(rec) + '\n')
    if s % a.log_every == 0 or s <= 3 or s == N:
        fp.flush()
        sj.flush()
        say('  step %5d  loss %.4f  bce %.4f  rho_hp %s (%d)  | p_shuf %.4f pen %.4f total %.4f rho_hp_shuf %s  lr %.2e  |g| %.3f  %.3f s/step (wait %.3f)  peak %.0f MB%s'
            % (s, loss, bce, 'nan' if not nq_tot else '%.4f' % rho, nq_tot, p_shuf, pen, loss_total, 'nan' if not nq_tot else '%+.4f' % rho_s,
               lr_now, gn, dt, wait, rec['vram_peak_alloc_mb'], '' if cm is None else '  v1 match %s' % cm))
    if not a.smoke and s == 300:
        mt = float(np.mean(times[100:300]))
        proj = mt * N / 3600
        say('  PROJECTION: mean step time over steps 101-300 %.3f s -> %d steps = %.2f h for this arm (bar 6 h)' % (mt, N, proj))
        if proj > 6.0:
            note = os.path.join(a.out, 'STOP-PROJECTION.txt')
            open(note, 'w', encoding='utf-8').write(
                'student2-train.py %s arm %s: mean step time over steps 101-300 %.4f s projects %d steps to %.2f h (> 6 h).\n'
                'prereg 3_recipe time_estimate: the run stops; a dated note beside the prereg scales the step count to 4 h for BOTH arms:\n'
                '  4 h / %.4f s = %d steps (written before any validation or test number exists).\n'
                % (time.strftime('%Y-%m-%dT%H:%M:%S%z'), ARM, mt, N, proj, mt, int(4 * 3600 / mt)))
            stop('projected %.2f h > 6 h; wrote %s' % (proj, note))
    if s in KEEP:
        ck = os.path.join(a.out, 'ckpt-LOGONLY-step%06d.pth' % s)
        torch.save({'model': {k: v.detach().cpu().float() for k, v in model.state_dict().items()}, 'config': base_cfg, 'step': s}, ck)
        say('  kept %s for the log (never scored, never selected) sha256 %s' % (ck, sha256(ck)))
    if a.val_every and s % a.val_every == 0 and s != N:
        model.eval()
        run_val(s)
        model.train()
fp.close()
sj.close()
model.eval()
run_val(N)
peak_a, peak_r = torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20
say('')
say('DONE %d steps; hp term zero on %d steps; loss first 10 mean %.4f, last 10 mean %.4f' % (N, hp_zero, float(np.mean(losses[:10])), float(np.mean(losses[-10:]))))
say('  v2 (logged only): p_shuf first 10 mean %.4f, last 10 mean %.4f; pen (LAMBDA %g) first 10 %.4f, last 10 %.4f; rho_hp_shuf first 10 %+.4f, last 10 %+.4f; permutation redraws %d'
    % (float(np.mean(pshufs[:10])), float(np.mean(pshufs[-10:])), LAM, LAM * float(np.mean(pshufs[:10])), LAM * float(np.mean(pshufs[-10:])),
       float(np.nanmean(rhoshufs[:10])), float(np.nanmean(rhoshufs[-10:])), NREDRAW[0]))
say('  clean-stream check (b): %s' % ('not run (smoke)' if REF is None else '%d of %d steps compared with v1\'s training log, all equal' % (n_match, N)))
skip = min(10, max(0, N - 10))
mt = float(np.mean(times[skip:]))
say('  step time: mean %.3f s over steps %d-%d (median %.3f, max %.3f); first step %.3f s' % (mt, skip + 1, N, float(np.median(times[skip:])), max(times[skip:]), times[0]))
say('  VRAM: torch peak allocated %.0f MB, peak reserved %.0f MB; nvidia-smi (used MiB, total, util%%) %s' % (peak_a, peak_r, smi_log))
say('  extrapolated to the prereg: %d steps x %.3f s = %.2f GPU h per arm, %.2f h for both arms' % (STEPS, mt, STEPS * mt / 3600, 2 * STEPS * mt / 3600))
if probe:
    pa = run_probe()
    say('  PROBE after: loss %.4f bce %.4f rho_hp %.4f (%d)  [before: loss %.4f bce %.4f rho_hp %.4f]'
        % (pa['loss'], pa['bce'], pa['rho_hp'], pa['n_rho'], pb['loss'], pb['bce'], pb['rho_hp']))
final = os.path.join(a.out, '%s%s2-final-step%06d.pth' % ('SMOKE-' if a.smoke else '', ARM, N))
sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
torch.save({'model': sd, 'config': base_cfg, 'step': N}, final)
fh = sha256(final)
say('  wrote %s sha256 %s (the %s: the weights after step %d, no early stopping)' % (final, fh, ARM, N))
# the payload must rebuild the way infer.py rebuilds the base
pl = torch.load(final, map_location='cpu', weights_only=True)
key, sd2 = extract_state_dict_entry_from_payload(pl, final)
m2 = make_model(normalize_training_config_for_inference(pl['config']))
inc2 = m2.load_state_dict(sd2, strict=True)
m2 = m2.to(dev).eval()
xt = torch.from_numpy(normalize_robust(window(names[0], *divmod(int(data[names[0]]['origins'][0]), data[names[0]]['Wo']), 0, 2)[0]))[None, None].to(dev)
with torch.no_grad():
    dmax = float((m2(xt)['ink'] - model(xt)['ink']).abs().max())
say('  reload as infer.py does (payload key %r, make_model(config)): missing %d, unexpected %d, config identical to the base %s, max |logit diff| %.3g'
    % (key, len(inc2.missing_keys), len(inc2.unexpected_keys), pl['config'] == base_cfg, dmax))
say('  guard: checked %d opens/listings, refused %d (its probe); twin a valid null on every segment: %s' % (_GUARD['checked'], _GUARD['refused'], twin_null))
say('  %.0f s' % (time.time() - T0))
runfile.done(REP, '\n'.join(lines) + '\n')
