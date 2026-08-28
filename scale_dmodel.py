import sys, argparse, json
sys.path.insert(0, '.')
import torch
from models import QISARTF, Transformer, Informer, Reformer, iTransformer, PatchTST, EDformer, DLinear, LightTS, MICN, Autoformer, Crossformer

def build_configs(**overrides):
    cfg = argparse.Namespace(
        task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=96,
        enc_in=7, dec_in=7, c_out=7, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3, distil=True, dropout=0.1, embed='timeF',
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

MODELS = {
    "QISARTF (FullAttn)": (QISARTF, dict(use_quantum_attention=False)),
    "QISARTF (QISA)": (QISARTF, dict(use_quantum_attention=True)),
    "iTransformer": (iTransformer, {}),
    "Transformer": (Transformer, {}),
    "PatchTST": (PatchTST, {}),
    "Informer": (Informer, {}),
}

results = {}
for d_model in [128, 256, 512, 1024]:
    d_ff = d_model * 4
    for name, (cls, overrides) in MODELS.items():
        cfg = build_configs(d_model=d_model, d_ff=d_ff, **overrides)
        try:
            model = cls.Model(cfg)
            n = sum(p.numel() for p in model.parameters())
            results.setdefault(name, []).append((d_model, n))
            print(f"d_model={d_model:5d}  {name:22s}  {n:>12,d} params")
        except Exception as e:
            print(f"d_model={d_model:5d}  {name:22s}  FAILED: {e}")

with open('scale_dmodel_results.json', 'w') as f:
    json.dump(results, f)
