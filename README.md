# Transformer-based Multi-step Time Series Forecasting of Methane Yield in Full-Scale Anaerobic Digestion
This is implementation code for methane yield forecasting model in the full-scale anaerobic digestion process


## Getting Started
1. Install requirements. `pip install -r requirements.txt`

2. Prepare the dataset 
* Create a folder named dataset/ in the project root.

* Convert your raw data into a CSV file (e.g., ad_data.csv).

* Ensure the CSV has:

    * A Date column (YYYY-MM-DD) as the index or first column.

    * Other columns with numerical values for each feature (e.g., HRTd, Qin_FW, Qin_PS, temp, etc.).

* Place the CSV file in dataset/.
3. Training & Evaluation
* Baseline models (RNN, LSTM, Vanilla Transformer):
```
sh scripts/main/Transformer.sh  # for vanilla Transformer
sh scripts/main/LSTM.sh         # for LSTM baseline
sh scripts/main/RNN.sh          # for RNN baseline
```
* Proposed model
```
sh scripts/main/ADFormer.sh
```
## Model Arguments
When running any of the training scripts, you can specify the following key arguments:
1. Feature dimensions (`--enc_in`,`--dec_in`,`--c_out`):
* ADFormer (proposed model):
    * Set `--enc_in`,`--dec_in`, and `--c_out `equal to the total number of input features.
* Other baselines (RNN, LSTM, Vanilla Transformer):
    * Set `--enc_in` and `--dec_in` to the number of features
    * Set `--c_out` to the number of target variables (here, 1 for methane yield).

2. Decoder mode (`--decoder_mode`):
* `default` -> used for forecasting (all future inputs masked)
* `future` -> used for simulation (provide actual future substrate inputs)
    * set `--n_subs` to the number of substrate-related features to unmask in the decoder


### Acknowledgement
We appreciate the following github repositories for their code base:
https://github.com/yuqinie98/PatchTST
https://github.com/zhouhaoyi/Informer2020


