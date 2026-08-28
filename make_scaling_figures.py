import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('scale_dmodel_results.json') as f:
    dmodel_res = json.load(f)
with open('scale_tokens_results.json') as f:
    token_res = json.load(f)

# --- Figure 1: parameter count vs d_model ---
fig, ax = plt.subplots(figsize=(6.5, 5))
colors = {"QISARTF (FullAttn)": "#1f77b4", "QISARTF (QISA)": "#d62728", "iTransformer": "#2ca02c",
          "Transformer": "#9467bd", "PatchTST": "#ff7f0e", "Informer": "#8c564b"}
for name, vals in dmodel_res.items():
    ds, ns = zip(*vals)
    ax.plot(ds, ns, 'o-', label=name, color=colors.get(name))
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel('$d_{model}$')
ax.set_ylabel('Parameter count')
ax.set_title('Parameter count scaling with model width')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_scale_dmodel.png', dpi=200)
print("saved fig_scale_dmodel.png")

# --- Figure 2: runtime vs N (reverse-embedding family) ---
fig2, ax2 = plt.subplots(figsize=(6.5, 5))
for name in ["QISARTF (FullAttn)", "QISARTF (QISA)", "iTransformer", "EDformer"]:
    vals = token_res[name]['N']
    ns, ts = zip(*vals)
    ax2.plot(ns, [t*1000 for t in ts], 'o-', label=name, color=colors.get(name, None))
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('Number of variates N')
ax2.set_ylabel('Forward+backward time (ms)')
ax2.set_title('Runtime scaling: reverse-embedding models vs. N')
ax2.legend(fontsize=8)
ax2.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_scale_N.png', dpi=200)
print("saved fig_scale_N.png")

# --- Figure 3: runtime vs L (timestep-tokenizing family) ---
fig3, ax3 = plt.subplots(figsize=(6.5, 5))
for name in ["Transformer", "Informer", "Autoformer", "PatchTST"]:
    vals = token_res[name]['L']
    ls, ts = zip(*vals)
    ax3.plot(ls, [t*1000 for t in ts], 'o-', label=name)
ax3.set_xscale('log'); ax3.set_yscale('log')
ax3.set_xlabel('Sequence length L')
ax3.set_ylabel('Forward+backward time (ms)')
ax3.set_title('Runtime scaling: timestep-tokenizing models vs. L')
ax3.legend(fontsize=8)
ax3.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_scale_L.png', dpi=200)
print("saved fig_scale_L.png")
