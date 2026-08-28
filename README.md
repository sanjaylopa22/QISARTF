# QISARTF: Quantum-Inspired Self-Attention Reverse Transformer for Time Series Analysis

QISARTF is a quantum-classical hybrid attention module embedded into an reversed time series transformer architecture which is suitable for deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, anomaly detection, and classification.**

Usage
Install Python 3.8. For convenience, execute the following command.
pip install -r requirements.txt
Prepare Data.

The following libraries are used as a core in this framework.

Time-Series-Library (TSlib)
TSlib is an open-source library for deep learning researchers, especially deep time series analysis.

## Datasets

The datasets are available at this [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) in the long-term-forecast folder. Download and keep them in the `dataset` folder here. 

Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder ./scripts/. You can reproduce the experiment results as the following examples:
# long-term forecast
bash ./scripts/long_term_forecast/Exchange_script/QISARTF.sh
# short-term forecast
bash ./scripts/short_term_forecast/M4/QISARTF_M4.sh

bash ./scripts/short_term_forecast/PEMS/QISARTF_PEMS.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/QISARTF.sh
# classification
bash ./scripts/classification/QISARTF.sh

Add the model file to the folder ./models. You can follow the ./models/Transformer.py.
Include the newly added model in the Exp_Basic.model_dict of ./exp/exp_basic.py.
Create the corresponding scripts under the folder ./scripts.

reproducibility/
├── model/               QISARTF.py, SelfAttention_Family.py, Embed.py, Transformer_EncDec.py, run.py
├── shell_scripts/       all QISARTF_*.sh training/eval scripts
├── verification/        count_params.py, gather_full_data.py, make_figures_v2.py
├── scaling_analysis/    scale_dmodel.py, scale_tokens.py, make_scaling_figures.py
├── quantum_circuit/     the n_q scaling benchmark, pennylane_qisa.py
└── README.md

# Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

# Current:
Sanjay Chakraborty (Postdoc, sanjay.chakraborty@liu.se)
Or describe it in Issues.
