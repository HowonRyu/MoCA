"""Export real MoCA reconstructions for the project-page demo -> docs/demo-data.js

Run from the repo root:  python docs/export_demo.py
Every curve in the demo is a forward pass of checkpoints/checkpoint-19.pth on the UCI-HAR test set,
under the four imputation settings of the paper (Table 3):
  random        70% of the 60 (sensor x 0.4 s) patches hidden at random positions
  temporal      2 consecutive time patches (20%, 0.8 s) hidden across all sensors (the paper's Table 3 hides 70%)
  sensor        one sensor fully hidden, reconstructed from the other five (the paper's Table 3 keeps one instead)
  extrapolation the last time patch (10%, 0.4 s) hidden across all sensors (the paper's Table 3 hides the last 70%)
"""
import sys, types, math, json
import numpy as np
import torch

# ponytail: shims so the repo code runs on torch 2.x / timm 1.x without editing it
sys.modules['torch._six'] = types.SimpleNamespace(inf=math.inf)
import timm.models.vision_transformer as tv
class _Block(tv.Block):
    def __init__(self, *a, qk_scale=None, **k): super().__init__(*a, **k)
tv.Block = _Block
sys.path.insert(0, '.')
from models_mae import mae_vit_base_patch16

CH = ['Gyro X', 'Gyro Y', 'Gyro Z', 'Acc X', 'Acc Y', 'Acc Z']  # 0-2 gyroscope, 3-5 accelerometer (gravity on 3-5)
FOCUS = 3          # Acc X is drawn large on the page
C, P, R = 6, 10, 0.7   # sensors, 20-step patches per sensor, mask ratio
CLASSES = {1: 'Walking', 2: 'Upstairs', 3: 'Downstairs', 0: 'Transition'}

model = mae_vit_base_patch16(img_size=[6, 200], patch_size=[1, 20], in_chans=1, alt=True)
ck = torch.load('checkpoints/checkpoint-19.pth', map_location='cpu', weights_only=False)['model']
missing, unexpected = model.load_state_dict(ck, strict=False)
assert not unexpected and not [k for k in missing if 'pos_embed' not in k], (missing, unexpected)  # pos_embed: fixed sin-cos
model.eval()

X = torch.tensor(np.asarray(torch.load('data/200/X_test_all.pt', weights_only=False)), dtype=torch.float32)
X = X.permute(0, 3, 2, 1).permute(0, 3, 1, 2)  # same as util/datasets.py UCIHAR(alt=True): (N, 1, 6, 200)
y = np.asarray(torch.load('data/200/y_test_all_mode.pt', weights_only=False))


def mask(task, v):
    """(6, 10) mask, 1 = hidden. v: random -> generator, temporal -> start patch, sensor -> kept sensor."""
    m = torch.zeros(C, P)
    if task == 'random':
        m.view(-1)[torch.randperm(C * P, generator=v)[:C * P - int(C * P * (1 - R))]] = 1
    elif task == 'temporal':
        m[:, v:v + 2] = 1  # ponytail: short gap for a readable demo
    elif task == 'sensor':
        m[v] = 1
    elif task == 'extrapolation':
        m[:, P - 1:] = 1  # ponytail: short horizon for a readable demo
    return m


@torch.no_grad()
def recon(x, masks):  # masks: (B, 6, 10), equal hidden count per row (true for every task above)
    _, pred, _ = model(x, import_mask=masks.flatten(1))
    return model.unpatchify(pred)[:, 0]  # (B, 6, 200)


# ---- test-set check against Table 3 (error on hidden values only) ----
g = torch.Generator().manual_seed(0)
draw = {'random': lambda: mask('random', g), 'temporal': lambda: mask('temporal', int(torch.randint(0, P - 1, (1,), generator=g))),
        'sensor': lambda: mask('sensor', int(torch.randint(0, C, (1,), generator=g))), 'extrapolation': lambda: mask('extrapolation', 0)}
stats = {}
for task, f in draw.items():
    se = ae = n = 0
    for i in range(0, len(X), 128):
        xb = X[i:i + 128]
        mb = torch.stack([f() for _ in range(len(xb))])
        hid = mb.repeat_interleave(20, dim=2).bool()  # (B, 6, 200)
        err = recon(xb, mb) - xb[:, 0]
        se += (err[hid] ** 2).sum().item(); ae += err[hid].abs().sum().item(); n += hid.sum().item()
    stats[task] = {'mse': round(se / n, 3), 'mae': round(ae / n, 3)}
    print(task, stats[task])

# ---- classification: fine-tuned MoCA (checkpoints/MoCAwAug_FT_checkpoint-49.pth) ----
import models_vit
NAMES = ['Transition', 'Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
clf = models_vit.vit_base_patch16(img_size=[6, 200], patch_size=[1, 20], num_classes=7, in_chans=1)
missing, unexpected = clf.load_state_dict(torch.load('checkpoints/MoCAwAug_FT_checkpoint-49.pth', map_location='cpu',
                                                     weights_only=False)['model'], strict=False)
assert not missing and not unexpected, (missing, unexpected)
clf.eval()
with torch.no_grad():
    prob = torch.cat([clf.head(clf.forward_features(X[i:i + 128])).softmax(1) for i in range(0, len(X), 128)]).numpy()
acc = float((prob.argmax(1) == y).mean())
print(f'test accuracy {acc:.4f} on {len(y)} windows')
# showcase: the 3 most confident correct test windows per class
top = lambda c, k: [int(i) for i in sorted(np.where((y == c) & (prob.argmax(1) == c))[0], key=lambda i: -prob[i, c])[:k]]
windows = [{'label': int(c), 'index': i, 'x': X[i, 0].double().numpy().round(3).tolist(), 'p': prob[i].round(3).tolist()}
           for c in [1, 2, 3, 4, 5, 6, 0] for i in top(c, 3)]
cls = {'names': NAMES, 'acc': round(acc * 100, 2), 'n': len(y), 'windows': windows}


def nerr(pred, x, m):
    """error on hidden values, each sensor scaled by its own variance in the window -> (B,)"""
    hid = m.repeat_interleave(20, dim=-1).expand_as(x).bool()
    e = (pred - x) ** 2 / x.var(-1, keepdim=True).clamp_min(1e-3)
    return (e * hid).sum((1, 2)) / hid.sum((1, 2))


# ---- imputation demo: per class, the window MoCA reconstructs best across all four tasks ----
variants = {'random': [torch.Generator().manual_seed(s) for s in range(5)], 'temporal': [0, 2, 4, 6, 8],
            'sensor': list(range(C)), 'extrapolation': [0]}  # sensor variant k hides sensor k
pool = np.where(np.isin(y, list(CLASSES)))[0]
xp = X[pool]
masks = {t: [mask(t, v) for v in vs] for t, vs in variants.items()}
errs = {t: torch.stack([nerr(recon(xp, torch.stack([m] * len(xp))), xp[:, 0], m) for m in ms]) for t, ms in masks.items()}  # (V, n)
score = sum(e.mean(0) / e.mean() for e in errs.values())  # each task weighted equally
samples = []
for cls_id, name in CLASSES.items():
    j = int(torch.tensor(np.where(y[pool] == cls_id)[0])[score[y[pool] == cls_id].argmin()])
    i, xi, tasks = int(pool[j]), X[pool[j]:pool[j] + 1], {}
    for t, ms in masks.items():
        order = list(range(len(ms))) if t == 'sensor' else errs[t][:, j].argsort().tolist()  # best mask first
        tasks[t] = [{'mask': ''.join(str(int(b)) for b in ms[k].flatten().tolist()),
                     'y': recon(xi, ms[k][None])[0].double().numpy().round(3).tolist()} for k in order]
    samples.append({'label': name, 'index': i, 'x': X[i, 0].double().numpy().round(3).tolist(), 'tasks': tasks})
    print(name, i, 'score', round(float(score[j]), 3))

# ---- "how it works" animation: a confidently classified walking window, 75% cross-modality mask (pre-training ratio) ----
def mask75(gen):
    m = torch.zeros(C * P); m[torch.randperm(C * P, generator=gen)[:C * P - int(C * P * 0.25)]] = 1
    return m.view(C, P)
best = None
for i in top(1, 10):
    for s in range(8):
        m = mask75(torch.Generator().manual_seed(s))
        pred = recon(X[i:i + 1], m[None])
        e = float(nerr(pred, X[i:i + 1, 0], m))
        if best is None or e < best[0]: best = (e, i, m, pred)
e, i, m, pred = best
how = {'index': i, 'x': X[i, 0].double().numpy().round(3).tolist(), 'mask': ''.join(str(int(b)) for b in m.flatten().tolist()),
       'y': pred[0].double().numpy().round(3).tolist(), 'p': prob[i].round(3).tolist(), 'label': int(y[i])}
print('how window', i, 'err', round(e, 3))
with torch.no_grad():  # the encoder's representation (CLS token) of this window; 48 of 768 dims for display
    z = model.forward_encoder(X[i:i + 1], mask_ratio=0.0)[0][0, 0]
z = z[torch.linspace(0, len(z) - 1, 48).long()]
how['z'] = (z / z.abs().max()).double().numpy().round(2).tolist()

with open('docs/demo-data.js', 'w') as f:
    f.write('window.DEMO = ' + json.dumps({'channels': CH, 'focus': FOCUS, 'stats': stats, 'samples': samples, 'cls': cls, 'how': how},
                                          separators=(',', ':')) + ';\n')
print('wrote docs/demo-data.js')
