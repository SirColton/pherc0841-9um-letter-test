"""day1-soup.py - single-trajectory weight soups of the released ink_9um checkpoints.

soup42 = uniform mean of seed42 steps 40k/50k/60k/75k; soup43 likewise for seed43. Never across
seeds: two seeds are two basins, and a cross-seed weight mean is not a model. The config and every
non-float tensor come from the step-75k file; the payload keeps koine infer's shape (model, config,
step) so infer.py rebuilds it unchanged. CPU only. Writes into geom/day1-9um/baselines/ only.
Usage: day1-soup.py
"""
import hashlib
import os

import torch

SRC = 'geom/ink9um'
OUT = 'geom/day1-9um/baselines'
STEPS = (40000, 50000, 60000, 75000)
os.makedirs(OUT, exist_ok=True)
for seed in (42, 43):
    paths = [os.path.join(SRC, 'seed%d-step-%06d.pth' % (seed, s)) for s in STEPS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit('missing step files: %s' % missing)
    payloads = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    keys = set(payloads[0]['model'])
    assert all(set(p['model']) == keys for p in payloads), 'state-dict keys differ within seed %d' % seed
    cfg = payloads[-1]['config']
    assert all(p['config'] == cfg for p in payloads), 'configs differ within seed %d' % seed
    soup = {}
    for k in payloads[-1]['model']:
        ts = [p['model'][k] for p in payloads]
        if ts[0].is_floating_point():
            soup[k] = torch.stack([t.double() for t in ts]).mean(0).to(ts[-1].dtype)
        else:
            soup[k] = ts[-1].clone()
    dmax = max(float((soup[k].double() - payloads[-1]['model'][k].double()).abs().max()) for k in soup if soup[k].is_floating_point())
    out = os.path.join(OUT, 'soup%d.pth' % seed)
    torch.save({'model': soup, 'config': cfg, 'step': 'soup-%s' % '-'.join(str(s) for s in STEPS)}, out)
    h = hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]
    print('soup%d: %d tensors from %s; max |soup - 75k| %.4g; wrote %s sha256 %s' % (
        seed, len(soup), ', '.join(os.path.basename(p) for p in paths), dmax, out, h))
