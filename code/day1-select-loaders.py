"""day1-select-loaders.py - loaders for the three MIT community checkpoints and the Scheirer reference arm.

Part "select" of the day-1 plan, prereg-v2 (geom/day1-9um/prereg-v2.json, sha256 checked below; a
mismatch stops the run). Each source file is checked against the sha256 prereg-v2 froze, read with
torch.load(weights_only=True) or safetensors (never pickle), and rewritten as the koine payload infer.py
rebuilds unchanged: {'model': state_dict, 'config': training config, 'step': n}.
  dense_native  Nieuwlaar: bare safetensors + train_dense_native.json (his card's three-line recipe)
  klavis_dense  domenicor046 dense9um-w016excluded-step075000: already {model, config, step}
  tauil_ft6k    TAUIL v1.0: weights under 'ema_model' (his ft9.py writes the plain fine-tuned
                state_dict there, "released format"); this koine build only accepts 'model', so the
                same tensors are moved to 'model'
  scheirer_ft   ShribyrLabs reader-ft-s42 step 12000: already {model, config, step}; REFERENCE ARM ONLY
Proof of load, all local: strict load_state_dict into koine's make_model(config) (0 missing, 0
unexpected), tensor and parameter counts (Nieuwlaar published 508 / 68,175,426), the inference-relevant
config fields against the released ink_9um seed43-75k, and weight-space distance to each card's stated
parent against an unrelated checkpoint as a control (a fine-tune must sit far nearer its parent).
No published benchmark number is reproduced here: every one needs data this agent may not download.
CPU only. Writes geom/day1-9um/select/ckpt/<name>.pth and geom/day1-9um/select/loaders.txt.
Usage: day1-select-loaders.py
"""
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import load_file

import runfile

KOINE = os.path.abspath('koine/ink-detection')
sys.path.insert(0, KOINE)
D9 = 'geom/day1-9um'
PREREG, PREREG_SHA = os.path.join(D9, 'prereg-v2.json'), '49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c'
CC = 'geom/ckpt-community'
OUT = os.path.join(D9, 'select', 'ckpt')
REP = os.path.join(D9, 'select', 'loaders.txt')
os.makedirs(OUT, exist_ok=True)
runfile.claim(REP)
lines = []


def say(s=''):
    print(s, flush=True)
    lines.append(s)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


got = sha(PREREG)
if got != PREREG_SHA:
    raise SystemExit('prereg-v2.json sha256 %s != frozen %s: stop' % (got, PREREG_SHA))
pre = json.load(open(PREREG, encoding='utf-8'))
frozen = dict(pre['members']['community_claim_eligible'])
frozen.update(pre['members']['reference_arm_only'])
say('day1-select-loaders.py  prereg-v2 sha256 %s (matches)' % got)

SPEC = [  # name, source, prereg key, arm
    ('dense_native', 'dense_native_016000.safetensors', 'dense_native (Nieuwlaar, MIT)', 'claim-eligible'),
    ('klavis_dense', 'dense9um-w016excluded-step075000.pth', 'KLAVIS dense w016excluded step75k (domenicor046, MIT per card)', 'claim-eligible'),
    ('tauil_ft6k', 'ink9um_native0139_ft6k.pth', 'TAUIL native0139 ft6k (MIT)', 'claim-eligible'),
    ('scheirer_ft', 'ink9um_ft_s42_step12000.pth', 'Scheirer ft_s42 step12000 (licence unstated)', 'reference arm only'),
]
REL = {'s42_75k': 'geom/ink9um/seed42-step-075000.pth', 's43_75k': 'geom/ink9um/seed43-step-075000.pth'}
CFG_KEYS = ('model_type', 'mode', 'in_channels', 'patch_size', 'image_normalization', 'mixed_precision', 'model_config', 'targets')

from koine_machines.inference.infer import normalize_training_config_for_inference, inference_preprocessing_from_config, inference_depth_from_config
from koine_machines.models.make_model import make_model

rel_cfg = torch.load(REL['s43_75k'], map_location='cpu', weights_only=True)['config']
weights = {k: torch.load(p, map_location='cpu', weights_only=True)['model'] for k, p in REL.items()}
say('released ink_9um parents/controls read weights_only: %s' % ', '.join('%s=%s' % kv for kv in REL.items()))
say('')
for name, src, pkey, arm in SPEC:
    path = os.path.join(CC, src)
    h = sha(path)
    ok = h == frozen[pkey]
    say('%s  [%s]  %s' % (name, arm, src))
    say('  sha256 %s  %s prereg-v2' % (h, 'matches' if ok else 'DIFFERS FROM'))
    if not ok:
        raise SystemExit('sha256 mismatch for %s: stop' % src)
    if src.endswith('.safetensors'):
        sd = load_file(path)
        cfg = json.load(open(os.path.join(CC, 'train_dense_native.json'), encoding='utf-8'))
        step, how = 16000, 'safetensors + train_dense_native.json (his card)'
    else:
        p = torch.load(path, map_location='cpu', weights_only=True)
        cfg = p['config']
        if 'model' in p:
            sd, step, how = p['model'], p['step'], "weights_only torch.load, payload['model']"
        else:
            sd, step, how = p['ema_model'], p.get('finetune', {}).get('step', p['step']), "weights_only torch.load, payload['ema_model'] -> 'model'"
            say('  finetune block: init %s, step %s, labels "%s"' % (
                os.path.basename(str(p['finetune'].get('init'))), p['finetune'].get('step'), p['finetune'].get('labels')))
    say('  read: %s; %d tensors, step %s' % (how, len(sd), step))
    diffs = {k: (cfg.get(k), rel_cfg.get(k)) for k in CFG_KEYS if cfg.get(k) != rel_cfg.get(k)}
    if 'model_config' in diffs and dict(cfg['model_config']) == dict(rel_cfg['model_config']):
        diffs.pop('model_config')
    say('  inference config vs released s43-75k: %s' % ('identical on %s' % ', '.join(CFG_KEYS) if not diffs else 'DIFFERS %s' % diffs))
    ncfg = normalize_training_config_for_inference(cfg)
    say('  infer.py will use: depth %d, preprocessing %s, amp %s' % (
        inference_depth_from_config(ncfg), inference_preprocessing_from_config(ncfg), cfg.get('mixed_precision')))
    model = make_model(ncfg)
    inc = model.load_state_dict(sd, strict=True)
    nparam = sum(t.numel() for t in sd.values() if t.is_floating_point())
    say('  strict load into make_model(config): missing %d, unexpected %d; float elements %d' % (
        len(inc.missing_keys), len(inc.unexpected_keys), nparam))
    if name == 'dense_native':
        say('  his VERIFY.md: 508 tensors, 68,175,426 parameters -> %s' % ('MATCH' if (len(sd), nparam) == (508, 68175426) else 'MISMATCH'))
    for rk, rw in weights.items():
        num = den = 0.0
        for k, t in sd.items():
            if t.is_floating_point() and k in rw:
                a, b = t.double(), rw[k].double()
                num += float(((a - b) ** 2).sum()); den += float((b ** 2).sum())
        say('  ||w - %s|| / ||%s|| = %.4f' % (rk, rk, (num / den) ** 0.5))
    if name == 'dense_native':
        kd = torch.load(os.path.join(CC, 'dense9um-w016excluded-step075000.pth'), map_location='cpu', weights_only=True)['model']
        num = sum(float(((sd[k].double() - kd[k].double()) ** 2).sum()) for k in sd if sd[k].is_floating_point())
        den = sum(float((kd[k].double() ** 2).sum()) for k in sd if sd[k].is_floating_point())
        say('  ||w - klavis_dense|| / ||klavis_dense|| = %.4f   (his stated init: KLAVIS dense ex016)' % ((num / den) ** 0.5))
    out = os.path.join(OUT, name + '.pth')
    torch.save({'model': sd, 'config': cfg, 'step': step}, out)
    say('  wrote %s sha256 %s' % (out, sha(out)))
    say('')
say('parents per card: tauil_ft6k and scheirer_ft are fine-tunes of released seed42 step-075000; klavis_dense is its')
say('  own seed-42 run (ink_9um recipe, dense labels); dense_native was initialised from KLAVIS dense ex016.')
runfile.done(REP, '\n'.join(lines) + '\n')
