# Industrial Machine Failure Prediction

**PBL Project — Section C**

## Overview
This project demonstrates a machine-learning workflow for predicting whether a machine is likely to experience failure based on operating measurements.

## Features used
- Machine Type (L, M, H)
- Air temperature (K)
- Process temperature (K)
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (min)

## Dataset
The repository includes `data/synthetic_machine_data.csv`, a **synthetic demonstration dataset** created for this project structure and code. It is not the original AI4I 2020 dataset and must not be represented as real industrial data. For the full project study, replace it with the AI4I 2020 Predictive Maintenance dataset and adapt column names as needed.

## Workflow
1. Data loading
2. Preprocessing (imputation, scaling, one-hot encoding)
3. Stratified train/test split
4. Random Forest classifier
5. 5-fold cross-validation and GridSearchCV
6. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
7. Save the trained pipeline and predict on a sample input

## Setup
```bash
pip install -r requirements.txt
python src/train_model.py
python src/predict.py
```

## Notes
- Dataset in this repository is synthetic and is for demonstration only.
- The target is `Machine failure` (0 = no failure, 1 = failure).
- Do not include failure-mode indicator columns as predictors; they can leak target information.
