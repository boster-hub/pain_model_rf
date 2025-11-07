# Random Forest (H2O) with SHAP Values and AUCPR Optimization

This repository contains an R script (`rf_h2o_shap.R`) for training and evaluating
Random Forest models using the **H2O** framework. It includes automated
hyperparameter tuning, repeated training iterations, feature importance
extraction, and SHAP value computation for interpretability.

---

## 🔍 Overview

The script automates the process of:
- Initializing an H2O environment  
- Performing a random grid search for Random Forest models  
- Optimizing for **Area Under the Precision-Recall Curve (AUCPR)**  
- Evaluating model performance across multiple iterations  
- Extracting feature importances and SHAP (Shapley) values  

This workflow is designed for scientific or applied machine-learning projects
where model stability and interpretability are important.

---

## ⚙️ Requirements

- **R ≥ 4.0**
- R packages:
  ```r
  install.packages(c("h2o", "caret", "dplyr", "ggplot2", "reshape2"))
  ```

---

## 🧠 Usage

### 1. Load your dataset

Load your data into R as a data frame named `input_data`, for example:

```r
input_data <- read.csv("pain_data.csv")
```

Make sure the column names in your dataset match the variable names you intend to use.

---

### 2. Define target and (optionally) feature subset

Set the name of your binary target variable.  
Optionally define a feature subset — if left `NULL`, the script uses all columns except the target.

```r
TARGET_VAR <- "op24"

# Optional: specify feature subset (otherwise uses all non-target columns)
FEATURE_VARS <- NULL
# or for example:
# FEATURE_VARS <- c("age", "sex_f")
```

---

### 3. Run the script

Run the R script after loading your data and setting variables.

```r
source("rf_h2o_shap.R")
```

The script will automatically:
- Preprocess the dataset  
- Start an H2O cluster  
- Perform Random Forest grid search with AUCPR optimization  
- Compute AUC, AUCPR, precision, recall, and SHAP values  
- Save performance plots (`auc_by_iteration.png`, `aucpr_by_iteration.png`)  
- Return all results as an R list called `results`

---

### 4. Access output

The object `results` (created by the script) contains the main results:

```r
# View aggregated AUC and AUCPR values across iterations
results$auc_results

# View feature importance results per iteration
results$importance_list[[1]]

# View SHAP values for one iteration
results$shap_list[[1]]

# Inspect metrics such as precision and recall
results$metrics_list[[1]]
```

---

## 🧩 Files in this repository

| File | Description |
|------|--------------|
| `rf_h2o_shap.R` | Main R script for model training and analysis |
| `README.md` | This documentation file |
| `LICENSE` | License information (MIT by default) |
| `data/` | (Optional) Folder for your dataset |

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE) — you’re free to use,
modify, and share it with attribution.

---

## 📚 Citation

If you use this code in your research, please cite:

> Jonas Henn (2025). **Random Forest (H2O) with SHAP and AUCPR Optimization**.  
> GitHub Repository: [https://github.com/boster-hub/pain_model_rf](https://github.com/boster-hub/pain_model_rf)

---
