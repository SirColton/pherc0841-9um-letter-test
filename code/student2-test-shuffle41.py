"""day1-test-shuffle41.py - the 41-layer depth-shuffled 252405 stack, a control input for day1-ink9-maps.py --stack.

WHY. The test part's task asks for "a depth-shuffled stack (permute the 41 layers with a fixed seed, logged)".
gates/prereg.json (controls_on_252405) pre-registered a different shuffle: each 21-layer slab permuted by
default_rng(0).permutation(21) before the rev flip (day1-ink9-maps.py --shuffle21 0). Both are run; this
script makes the input for the 41-layer one. The seed is 0, the prereg's own seed, so neither seed is picked
after looking at anything.

Writes <out>/252405-stack-shuf41.npy (same shape and dtype; stack[perm]) and <out>/shuf41.json (the
permutation, input and output sha256). CPU only, no model. Usage: day1-test-shuffle41.py <out dir>
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

SEED = 0
SRC = 'geom/day1-9um/render/252405-stack.npy'
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


st = np.load(SRC)
perm = np.random.default_rng(SEED).permutation(st.shape[0])
sh = np.ascontiguousarray(st[perm])
dst = os.path.join(OUT, '252405-stack-shuf41.npy')
np.save(dst, sh)
c0 = st.shape[0] // 2
info = {'written': time.strftime('%Y-%m-%dT%H:%M:%S'), 'source': SRC, 'source_sha256': sha256(SRC),
        'source_array_sha1_12': hashlib.sha1(st.tobytes()).hexdigest()[:12], 'shape': list(st.shape), 'dtype': str(st.dtype),
        'seed': SEED, 'rule': 'out[k] = stack[perm[k]], perm = numpy default_rng(0).permutation(41)', 'perm': perm.tolist(),
        'slab_source_layers': {'c%+d' % dc: perm[c0 + dc - 10:c0 + dc + 11].tolist() for dc in (-4, -2, 0, 2, 4)},
        'output': dst, 'output_sha256': sha256(dst)}
json.dump(info, open(os.path.join(OUT, 'shuf41.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != 'slab_source_layers'}, indent=1))
