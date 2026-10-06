"""day1-ink9-maps.py - ink_9um-format checkpoints on the cached w00big native 9.366 um stack.

Same recipe as c0841-9um.py's ink9(), so the maps are comparable with the cached ones: a 21-layer
window about depth centre c (stack middle 20 + c), reversed for 'rev', written as a zarr v2
(21, H, W) and read by koine's infer.py three times (layers 0-17, 2-19, 4-21); the map is the
mean of the three uint8 outputs, saved raw as float32. infer.py is called in-process (main(argv)),
so torch and the module load once; --num-workers 0 because this runs under polite-run.py, whose
module has no __main__ guard for spawned workers. GPU only through polite-run.py.

Usage: polite-run.py day1-ink9-maps.py <outdir> <name=ckpt> [<name=ckpt> ...] [--centres -4,0,4] [--orders rev,fwd] [--tta-mirror]
       [--stack <stack.npy>] [--shuffle21 <seed>]
--tta-mirror passes infer.py's mirror TTA through; the file names do not mark it, so give TTA runs their own names.
Writes <outdir>/<name>-c<+d>-<order>.npy, skips any that exist, and logs to <outdir>/maps-log.txt.
"""
import os
import shutil
import sys
import time

import numpy as np

KOINE = os.path.abspath('koine/ink-detection')
sys.path.insert(0, KOINE)
STACK = 'geom/l0-1203-render/audit/build-deciding-test/ours-stack.npy'
WINDOWS = ((0, 17), (2, 19), (4, 21))
CH = 128

args = sys.argv[1:]
centres, orders = (-4, -2, 0, 2, 4), ('rev',)
TTA = '--tta-mirror' in args
if TTA:
    args.remove('--tta-mirror')
if '--centres' in args:
    i = args.index('--centres'); centres = tuple(int(x) for x in args[i + 1].split(',')); del args[i:i + 2]
if '--orders' in args:
    i = args.index('--orders'); orders = tuple(args[i + 1].split(',')); del args[i:i + 2]
# --stack <npy>: another rendered stack (default the w00big one). --shuffle21 <seed>: depth control, each 21-layer
# slab permuted by default_rng(seed).permutation(21) before the rev flip (gates/prereg.json controls_on_252405).
if '--stack' in args:
    i = args.index('--stack'); STACK = args[i + 1]; del args[i:i + 2]
SH21 = None
if '--shuffle21' in args:
    i = args.index('--shuffle21'); SH21 = int(args[i + 1]); del args[i:i + 2]
outdir = args[0]
ckpts = [a.split('=', 1) for a in args[1:]]
os.makedirs(outdir, exist_ok=True)
tmp = os.path.abspath(os.path.join(outdir, 'tmp'))
os.makedirs(tmp, exist_ok=True)
log = open(os.path.join(outdir, 'maps-log.txt'), 'a', encoding='utf-8')


def say(s):
    print(s, flush=True)
    log.write(s + '\n'); log.flush()


import tifffile
import zarr
from PIL import Image
from koine_machines.inference import infer

stack = np.load(STACK, mmap_mode='r')
c0 = stack.shape[0] // 2
say('%s day1-ink9-maps: stack %s, centre index %d, centres %s, orders %s, ckpts %s%s' % (
    time.strftime('%Y-%m-%d %H:%M:%S'), stack.shape, c0, centres, orders, [n for n, _ in ckpts], ', mirror TTA' if TTA else ''))
say('  stack file %s' % os.path.abspath(STACK))
if SH21 is not None:
    say('  --shuffle21 %d: slab layer order %s (applied before the rev flip)' % (SH21, np.random.default_rng(SH21).permutation(21).tolist()))
t0 = time.time()
for dc in centres:
    c = c0 + dc
    s21 = np.ascontiguousarray(stack[c - 10:c + 11])
    if SH21 is not None:
        s21 = np.ascontiguousarray(s21[np.random.default_rng(SH21).permutation(21)])
    for o in orders:
        todo = [(n, p) for n, p in ckpts if not os.path.exists(os.path.join(outdir, '%s-c%+d-%s.npy' % (n, dc, o)))]
        if not todo:
            continue
        s = s21 if o == 'fwd' else np.ascontiguousarray(s21[::-1])
        zpath = os.path.join(tmp, 'in-c%+d-%s.zarr' % (dc, o))
        shutil.rmtree(zpath, ignore_errors=True)
        z = zarr.open(zpath, mode='w', shape=s.shape, chunks=(s.shape[0], CH, CH), dtype='u1', zarr_format=2)
        z[:] = s
        for n, p in todo:
            t1 = time.time()
            preds = []
            for (a, b) in WINDOWS:
                tif = os.path.join(tmp, 'pred-%s-%d_%d.tif' % (n, a, b))
                if os.path.exists(tif):
                    os.remove(tif)
                infer.main([zpath, os.path.abspath(p), tif, '--layer-start', str(a), '--layer-end', str(b),
                            '--no-compile', '--num-workers', '0'] + (['--tta-mirror'] if TTA else []))
                q = tifffile.imread(tif).astype(np.float32)
                q = q[0] if q.ndim == 3 else q
                if q.shape != s.shape[1:]:
                    q = np.asarray(Image.fromarray(q).resize(s.shape[1:][::-1], Image.LANCZOS), np.float32)
                preds.append(q)
                os.remove(tif)
            m = np.mean(preds, axis=0).astype(np.float32)
            np.save(os.path.join(outdir, '%s-c%+d-%s.npy' % (n, dc, o)), m)
            say('  %s c%+d %s: mean %.1f sd %.1f, %.0f s (total %.0f s)' % (n, dc, o, m.mean(), m.std(), time.time() - t1, time.time() - t0))
        shutil.rmtree(zpath, ignore_errors=True)
say('done %.0f s' % (time.time() - t0))
