import sys, argparse, json, time
sys.path.insert(0, '.')
import torch
from models import QISARTF, Transformer, Informer, Reformer, iTransformer, PatchTST, EDformer, DLinear, LightTS, Autoformer

def build_configs(**overrides):
    cfg = argparse.Namespace(
        task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=96,
        enc_in=7, dec_in=7, c_out=7, d_model=128, n_heads=8, e_layers=2, d_layers=1,
        d_ff=512, moving_avg=25, factor=3, distil=True, dropout=0.1, embed='timeF',
        freq='h', activation='gelu', output_attention=False, channel_independence=1,
        decomp_method='moving_avg', use_norm=1, down_sampling_layers=0,
        down_sampling_window=1, down_sampling_method=None, seg_len=48,
        top_k=5, num_kernels=6, expand=2, d_conv=4, num_class=0,
        p_hidden_dims=[128, 128], p_hidden_layers=2, use_quantum_attention=False,
        use_reverse_embedding=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

# Reverse-embedding family: token count = N (variates). Vary enc_in.
REV_MODELS = {
    "QISARTF (FullAttn)": (QISARTF, dict(use_quantum_attention=False)),
    "QISARTF (QISA)": (QISARTF, dict(use_quantum_attention=True)),
    "iTransformer": (iTransformer, {}),
    "EDformer": (EDformer, {}),
}
# Timestep-tokenizing family: token count = L (sequence length). Vary seq_len.
TS_MODELS = {
    "Transformer": (Transformer, {}),
    "Informer": (Informer, {}),
    "Autoformer": (Autoformer, {}),
    "PatchTST": (PatchTST, {}),
}

def time_forward_backward(model, x_enc, x_mark_enc, x_dec, x_mark_dec, n_reps=3):
    for _ in range(2):
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out.sum().backward()
        model.zero_grad()
    t0 = time.time()
    for _ in range(n_reps):
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out.sum().backward()
        model.zero_grad()
    return (time.time() - t0) / n_reps

results = {}
print("=== Reverse-embedding family: scaling with N (number of variates) ===")
for N in [7, 21, 96, 358]:
    cfg_base = dict(enc_in=N, dec_in=N, c_out=N)
    for name, (cls, overrides) in REV_MODELS.items():
        cfg = build_configs(**cfg_base, **overrides)
        try:
            model = cls.Model(cfg)
            x_enc = torch.randn(4, cfg.seq_len, N); x_mark = torch.randn(4, cfg.seq_len, 4)
            x_dec = torch.zeros(4, cfg.pred_len, N); x_mark_dec = torch.zeros(4, cfg.pred_len, 4)
            t = time_forward_backward(model, x_enc, x_mark, x_dec, x_mark_dec)
            results.setdefault(name, {}).setdefault('N', []).append((N, t))
            print(f"N={N:4d}  {name:20s}  {t*1000:8.2f} ms")
        except Exception as e:
            print(f"N={N:4d}  {name:20s}  FAILED: {str(e)[:60]}")

print()
print("=== Timestep-tokenizing family: scaling with L (sequence length) ===")
for L in [96, 192, 336, 720]:
    cfg_base = dict(seq_len=L, label_len=L//2, pred_len=96)
    for name, (cls, overrides) in TS_MODELS.items():
        cfg = build_configs(**cfg_base, **overrides)
        try:
            model = cls.Model(cfg)
            x_enc = torch.randn(4, L, 7); x_mark = torch.randn(4, L, 4)
            x_dec = torch.zeros(4, cfg.label_len + cfg.pred_len, 7)
            x_mark_dec = torch.zeros(4, cfg.label_len + cfg.pred_len, 4)
            t = time_forward_backward(model, x_enc, x_mark, x_dec, x_mark_dec)
            results.setdefault(name, {}).setdefault('L', []).append((L, t))
            print(f"L={L:4d}  {name:20s}  {t*1000:8.2f} ms")
        except Exception as e:
            print(f"L={L:4d}  {name:20s}  FAILED: {str(e)[:60]}")

with open('scale_tokens_results.json', 'w') as f:
    json.dump(results, f)
