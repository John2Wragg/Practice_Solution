# Hastings Solution

## Overview
- 

## Setup
```bash
uv sync
```
- 

## Repo Structure
- `Data_Preparation/`: 
- `functions/`: 
- `params/`: 
- `Model_Building/`: 
- `Model_Evaluation/`: 
- `Model_Experiments/`: 
- `Prepped_Data/`: 
- `Snips/`: 

## Data Preparation
- Contains `Dataloader.py` class which loads and transforms the raw data. Including train test split. Helper functions can be found in `functions.dataloader_helpers.py`. 

## Model Building
- Feature Selection CV: `run_rfe.py`:
invoked via `python run_rfe.py` Used the columns in the data, and removed the `DO_NOT_MODEL` chosen in params.py. 

Ran RFE from sklearn with cross validated samples on the training data to determine features. 

Saved into model experiments folder via run_name
- Optuna HPT trials ran for each of NIl, Severity and large loss models

Best params are saved into the experiments folder. . `run_hpt.py`.
- Contains `ModelBuilder.py` a class which holds each modelling component (nil, sev, large_loss) and handled the fit and predict cases for each. 

## Model Evaluation
- Calculates the metrics on the test dataset for each of the nil, severity and large loss modelbuilder components. Includes SHAP and AvE plots by feature. Also includes confusion matrix plots with threshold chosen from ROC. 
- `run_evaluation.py`
- In each stage a run_name arg can be passed. All outputs will be saved in a Model_Experiments folder within the repo. For a subsequent stage to run, the correct artefacts must be present as a result of running the prior stage with the corresponding run_name.  

## Large Loss Approach
- Added an additional component into the ModelBuilder class which models the P(Large Loss). The excess is given by filtering the train dataset to where large loss flag == 1, and take the mean (incurred - capped). 

