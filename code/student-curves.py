"""student-curves.py - loss curves of the student and its scramble twin, from the harness's own logs only.

Reads geom/student-9um/train-student/{steps,val}.jsonl and geom/student-9um/train-scramble/{steps,val}.jsonl
(student-train.py writes them; nothing else is read, no 252405 file, no model). Writes, under
geom/student-9um/curves/:
  loss-curves-100.csv   per arm, per 100-step bin: mean loss, bce, rho_hp (over steps where it was defined), grad
                        norm, step time; plus the count of steps whose hp term was zero
  loss-curves.svg       three panels (loss, bce, rho_hp), bin means; student solid blue, twin dashed orange, so the
                        line style carries the arm as well as the hue; the validation strip's own-target loss as
                        markers on the loss panel (logged only, chooses nothing)
  curves-run.txt        the summary numbers
The curves describe training; nothing here is scored, and nothing is chosen from them (prereg 3_recipe
early_stopping: NONE).
Usage (from <workdir>, CPU only): python student-curves.py
"""
import json
import os
import sys

import numpy as np

import runfile

ROOT = 'geom/student-9um'
ARMS = [('student', 'train-student', '#1f5fbf', ''), ('scramble', 'train-scramble', '#d9730d', '7,5')]
OUT = os.path.join(ROOT, 'curves')
BIN = 100
os.makedirs(OUT, exist_ok=True)
REP = os.path.join(OUT, 'curves-run.txt')
runfile.claim(REP)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def load(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


data = {}
for arm, d, _, _ in ARMS:
    st = load(os.path.join(ROOT, d, 'steps.jsonl'))
    va = load(os.path.join(ROOT, d, 'val.jsonl'))
    steps = np.array([r['step'] for r in st])
    if len(steps) == 0 or not np.array_equal(steps, np.arange(1, len(steps) + 1)):
        sys.exit('%s steps.jsonl is not steps 1..N in order' % arm)
    col = lambda k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in st])
    data[arm] = {'step': steps, 'loss': col('loss'), 'bce': col('bce'), 'rho': col('rho_hp'), 'g': col('grad_norm'),
                 'dt': col('step_s'), 'lr': col('lr'), 'nrho': col('n_rho'), 'val': va,
                 'in': [r['in_sha1'] for r in st], 'mask': [r['mask_sha1'] for r in st], 'tgt': [r['target_sha1'] for r in st]}

say('student-curves.py: loss curves from the harness logs (no 252405 file, no model, nothing scored or chosen)')
for arm, d, _, _ in ARMS:
    x = data[arm]
    n = len(x['step'])
    say('')
    say('%s (%s): %d steps logged' % (arm, d, n))
    say('  loss  first 100 mean %.4f   last 100 mean %.4f   min bin-of-100 mean %.4f' % (
        np.nanmean(x['loss'][:100]), np.nanmean(x['loss'][-100:]),
        min(np.nanmean(x['loss'][i:i + BIN]) for i in range(0, n, BIN))))
    say('  bce   first 100 mean %.4f   last 100 mean %.4f' % (np.nanmean(x['bce'][:100]), np.nanmean(x['bce'][-100:])))
    say('  rho_hp first 100 mean %.4f  last 100 mean %.4f   (steps with the hp term zero: %d)' % (
        np.nanmean(x['rho'][:100]), np.nanmean(x['rho'][-100:]), int(np.sum(x['nrho'] == 0))))
    say('  grad norm first 100 mean %.3f  last 100 mean %.3f; step time median %.3f s, mean over steps 11-%d %.3f s' % (
        np.nanmean(x['g'][:100]), np.nanmean(x['g'][-100:]), np.nanmedian(x['dt']), n, np.nanmean(x['dt'][10:])))
    say('  lr at steps 1, 500, 501, 6250, %d: %s' % (n, ', '.join('%.3g' % x['lr'][k - 1] for k in (1, 500, 501, 6250, n) if k <= n)))
    for v in x['val']:
        o = v['own']
        t = v.get('teacher')
        say('  VAL step %5d strip (logged only, chooses nothing): own target loss %.4f bce %.4f rho_hp %.4f (%d windows, %d px)%s' % (
            v['step'], o['loss'], o['bce'], o['rho_hp'], o['n_rho'], o['px'],
            '; vs the teacher bce %.4f rho_hp %.4f' % (t['bce'], t['rho_hp']) if t else ''))

if len(data['student']['step']) == len(data['scramble']['step']):
    same_in = sum(a == b for a, b in zip(data['student']['in'], data['scramble']['in']))
    same_m = sum(a == b for a, b in zip(data['student']['mask'], data['scramble']['mask']))
    same_t = sum(a == b for a, b in zip(data['student']['tgt'], data['scramble']['tgt']))
    n = len(data['student']['step'])
    say('')
    say('twin identity (steps.jsonl batch hashes): input equal on %d of %d steps, mask equal on %d of %d, target equal on %d of %d'
        % (same_in, n, same_m, n, same_t, n))

# ---- csv ------------------------------------------------------------------------------------------
rows = ['arm,bin_first_step,bin_last_step,loss_mean,bce_mean,rho_hp_mean,rho_steps,grad_norm_mean,step_s_mean,hp_zero_steps']
bins = {}
for arm, _, _, _ in ARMS:
    x = data[arm]
    n = len(x['step'])
    b = []
    for i in range(0, n, BIN):
        sl = slice(i, min(n, i + BIN))
        r = x['rho'][sl]
        b.append((i + 1, min(n, i + BIN), np.nanmean(x['loss'][sl]), np.nanmean(x['bce'][sl]),
                  np.nanmean(r) if np.isfinite(r).any() else np.nan, int(np.isfinite(r).sum()),
                  np.nanmean(x['g'][sl]), np.nanmean(x['dt'][sl]), int(np.sum(x['nrho'][sl] == 0))))
        rows.append('%s,%d,%d,%.6f,%.6f,%.6f,%d,%.5f,%.4f,%d' % ((arm,) + b[-1]))
    bins[arm] = b
open(os.path.join(OUT, 'loss-curves-100.csv'), 'w', encoding='utf-8').write('\n'.join(rows) + '\n')

# ---- svg ------------------------------------------------------------------------------------------
W, PH, ML, MR, MT, GAP = 900, 220, 70, 120, 40, 50
H = MT + 3 * PH + 2 * GAP + 50
NMAX = max(len(data[a]['step']) for a, _, _, _ in ARMS)
panels = [('loss', 2, 'loss = BCE + 0.5 (1 - rho_hp)'), ('bce', 3, 'BCE over training px'), ('rho', 4, 'rho_hp (48 um high-pass Pearson)')]
out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="sans-serif" font-size="12">' % (W, H, W, H),
       '<rect width="100%" height="100%" fill="#ffffff"/>',
       '<text x="%d" y="22" font-size="15" fill="#111">Student vs scramble twin: 100-step bin means (training curves only; nothing chosen from them)</text>' % ML]
for pi, (key, idx, title) in enumerate(panels):
    top = MT + pi * (PH + GAP)
    vals = [b[idx] for a, _, _, _ in ARMS for b in bins[a] if np.isfinite(b[idx])]
    if key == 'loss':
        vals += [v['own']['loss'] for a, _, _, _ in ARMS for v in data[a]['val'] if np.isfinite(v['own']['loss'])]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    X = lambda s: ML + (W - ML - MR) * s / NMAX
    Y = lambda v: top + PH - PH * (v - lo) / (hi - lo)
    out.append('<text x="%d" y="%d" font-size="13" fill="#111">%s</text>' % (ML, top - 8, title))
    out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#999"/>' % (ML, top, W - ML - MR, PH))
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#e4e4e4"/>' % (ML, W - MR, Y(v), Y(v)))
        out.append('<text x="%d" y="%.1f" text-anchor="end" fill="#444">%.3f</text>' % (ML - 6, Y(v) + 4, v))
    for s in range(0, NMAX + 1, 2000):
        out.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="#e4e4e4"/>' % (X(s), X(s), top, top + PH))
        out.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#444">%d</text>' % (X(s), top + PH + 15, s))
    for arm, _, colr, dash in ARMS:
        pts = [(X((b[0] + b[1]) / 2), Y(b[idx])) for b in bins[arm] if np.isfinite(b[idx])]
        if not pts:
            continue
        out.append('<polyline fill="none" stroke="%s" stroke-width="2"%s points="%s"/>' % (
            colr, ' stroke-dasharray="%s"' % dash if dash else '', ' '.join('%.1f,%.1f' % p for p in pts)))
        out.append('<text x="%.1f" y="%.1f" fill="%s" font-weight="%s">%s</text>' % (
            pts[-1][0] + 6, pts[-1][1] + 4 + (12 if arm == 'scramble' else 0), colr, 'bold' if arm == 'student' else 'normal', arm))
        if key == 'loss':
            for v in data[arm]['val']:
                if np.isfinite(v['own']['loss']):
                    cx, cy = X(v['step']), Y(v['own']['loss'])
                    if arm == 'student':
                        out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"/>' % (cx, cy, colr))
                    else:
                        out.append('<rect x="%.1f" y="%.1f" width="8" height="8" fill="none" stroke="%s" stroke-width="2"/>' % (cx - 4, cy - 4, colr))
out.append('<text x="%d" y="%d" fill="#444">training step. Markers on the loss panel: validation-strip loss against the arm\'s own target '
           '(student filled circles, twin open squares), logged every 2,000 steps, chooses nothing.</text>' % (ML, H - 12))
out.append('</svg>')
open(os.path.join(OUT, 'loss-curves.svg'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
say('')
say('wrote %s/loss-curves-100.csv and %s/loss-curves.svg' % (OUT, OUT))
runfile.done(REP, '\n'.join(lines) + '\n')
