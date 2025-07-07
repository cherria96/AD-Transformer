# Transformer-based Multi-step Time Series Forecasting of Methane Yield in Full-Scale Anaerobic Digestion
This is implementation code for methane yield forecasting model in the full-scale anaerobic digestion process


## Getting Started
1. Install requirements. `pip install -r requirements.txt`

2. Create a separate folder `./dataset` and put csv files in the directory

3. Training: run the scripts in the directory `./scripts/`

### Baseline models: RNN, LSTM, Vanilla Transformer
e.g. if you want to train vanilla Transformer model
```
sh scripts/sbk_ad/Transformer.sh
```
### Proposed model
```
sh scripts/sbk_ad/ADFormer.sh
```


