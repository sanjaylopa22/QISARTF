# QISARTF: Quantum-Inspired Self-Attention Reverse Transformer for Time Series Analysis

QISARTF embeds a quantum-inspired self-attention (QISA) module into a reverse time-series transformer. QISA replaces the classical dot-product attention score with a construction built directly from the mathematics of quantum mechanics: queries and keys are mapped to normalized complex-valued states in a Hilbert space, evolved by a trainable unitary operator, and combined into an attention score that is a genuine Born-rule transition probability. This is a classical computation of quantum-mechanical mathematics, not execution on quantum hardware and not a hybrid quantum-classical circuit — a separate circuit-based variant of QISA (Section on hybrid QISA / `quantum_circuit/`) is analyzed for comparison but is not the mechanism used to produce this paper's main results.

The codebase supports four tasks: **long-term forecasting, short-term forecasting, anomaly detection, and classification.** Imputation is not implemented (`QISARTF.py`'s `forward()` raises `NotImplementedError` for this task) and should not be attempted with this model.

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`. Optional, needed only for the reproducibility/analysis scripts described below: `matplotlib` (figures), `pennylane==0.45.1` (hybrid quantum-circuit analysis), `einops` and `reformer_pytorch` (needed to import certain baseline models used for comparison, e.g. Crossformer and Reformer).

## Datasets

The datasets are available at this [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) in the long-term-forecast folder. Download and keep them in the `dataset` folder here. 

## Training and Evaluation

Scripts are provided for both the classical baseline (`FullAttention`) and the QISA-enabled variant of every experiment. Running a script trains and tests the model as specified; the `_QISA` suffix on a script name enables `--use_quantum_attention`, and both variants write to distinct checkpoint directories (`model_id` is suffixed with `_QISA` in the QISA scripts) so the two do not overwrite each other's results.

```bash
# Long-term forecasting
bash ./scripts/long_term_forecast/QISARTF_ETTh1.sh
bash ./scripts/long_term_forecast/QISARTF_ETTh1_QISA.sh
bash ./scripts/long_term_forecast/QISARTF_Exchange.sh
bash ./scripts/long_term_forecast/QISARTF_Exchange_QISA.sh

# Short-term forecasting
bash ./scripts/short_term_forecast/QISARTF_M4.sh
bash ./scripts/short_term_forecast/QISARTF_M4_QISA.sh
bash ./scripts/short_term_forecast/QISARTF_PEMS.sh
bash ./scripts/short_term_forecast/QISARTF_PEMS_QISA.sh

# Anomaly detection (SMD, MSL, SMAP, SWaT, PSM)
bash ./scripts/anomaly_detection/QISARTF_anomaly_SMD.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_SMD_QISA.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_MSL.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_MSL_QISA.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_SMAP.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_SMAP_QISA.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_SWaT.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_SWaT_QISA.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_PSM.sh
bash ./scripts/anomaly_detection/QISARTF_anomaly_PSM_QISA.sh

# Classification (10 UEA datasets in one script)
bash ./scripts/classification/QISARTF_classification_UEA.sh
bash ./scripts/classification/QISARTF_classification_UEA_QISA.sh
```

`[NOTE: the paths above assume you place each script under a `scripts/<task>/` subfolder matching the task it runs, mirroring TSlib's layout. The 20 scripts as provided are flat files; reorganize into this structure, or adjust the paths above to match wherever you actually place them, before distributing.]`

## Reproducibility Package

In addition to the model and training scripts, this release includes the scripts used to generate every table, figure, and numeric claim in the paper's verification, parameter-count, scaling, and quantum-circuit-analysis sections. Several of these scripts are also what caught and fixed real bugs during development (see Known Issues below) — they are included as-is, not cleaned up to hide that history.

```
reproducibility/
├── model/
│   ├── QISARTF.py                  Main model (forecast/anomaly_detection/classification)
│   ├── SelfAttention_Family.py     QuantumAttention (UnitaryQISA) and FullAttention
│   ├── Embed.py                    DataEmbedding, DataEmbedding_inverted
│   ├── Transformer_EncDec.py       Encoder / EncoderLayer
│   └── run.py                      CLI entry point (--use_quantum_attention, --use_reverse_embedding, etc.)
├── shell_scripts/
│   └── QISARTF_*.sh, QISARTF_*_QISA.sh   All 20 training/eval scripts described above
├── verification/
│   ├── count_params.py             Parameter counts for QISARTF vs. baseline models
│   ├── gather_full_data.py         Module-level attention timing (QISA vs. FullAttention)
│   └── make_figures_v2.py          Builds the attention-scaling and slowdown-ratio figures
├── scaling_analysis/
│   ├── scale_dmodel.py             Parameter count vs. model width (d_model)
│   ├── scale_tokens.py             Runtime vs. token count (variates N and sequence length L)
│   └── make_scaling_figures.py     Builds the corresponding figures
├── quantum_circuit/
│   ├── pennylane_qisa.py           Hybrid circuit-based QISA variant (not the deployed mechanism)
│   └── nq_scaling_benchmark.py     Gate count, circuit depth, and n_q scaling for the hybrid variant
└── README.md
```

### Notes on reproducing the reported numbers

- **Absolute timings will not match exactly.** All timing figures in the paper's scaling and verification sections were measured on CPU in a sandboxed environment, not the GPU hardware used for the main training runs. Re-running these scripts on different hardware will change the absolute millisecond values; the *relative* patterns they demonstrate (QISA's per-token overhead vs. classical attention, Transformer's near-quadratic growth vs. sub-quadratic baselines, exponential classical simulation cost vs. qubit count) should hold regardless of hardware.
- **Package versions matter for the quantum-circuit scripts specifically.** `qml.specs()`'s reported gate counts and circuit depth are extracted directly from PennyLane's own circuit compiler; results were generated with `pennylane==0.45.1` and may differ on other versions.
- **Import paths assume a specific package layout** (`models/`, `layers/`, `utils/` importable from the working directory). Adjust `sys.path` at the top of each script if your repository layout differs.
- **The fidelity/noise projection in the quantum-circuit analysis is illustrative, not measured** — we do not have access to physical quantum hardware; it uses representative gate-fidelity figures from published hardware literature, applied to our circuit's gate counts, and is explicitly labeled as such in the paper text.

## Extending This Work

To add a new attention mechanism or model variant:
1. Add the model file to `./models/`, following the structure of `./models/QISARTF.py` or `./models/Transformer.py`.
2. Register it in `Exp_Basic.model_dict` in `./exp/exp_basic.py`.
3. Add a corresponding training/eval script under `./scripts/`.

## Contact

If you have any questions or suggestions, feel free to contact our maintenance team:

**Current:** Sanjay Chakraborty (Postdoc, sanjay.chakraborty@liu.se)

Or open an Issue.
