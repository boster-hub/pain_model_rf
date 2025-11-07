# rf_h2o_shap.R
# -------------------------------------------------------------------
# Random Forest (H2O) with AUCPR-tuned random search, feature importance,
# SHAP values, and repeated train/test iterations.
#
# Author: <your name>
# Created: 2025-11-07
# -------------------------------------------------------------------
# Usage:
#   1) Load your dataset into R as `input_data`
#      (e.g., input_data <- read.csv("pain_data.csv"))
#   2) Set TARGET_VAR below (binary 0/1 outcome expected).
#   3) Run: source("rf_h2o_shap.R")
#
# Notes:
#   - If FEATURE_VARS is NULL, all columns except TARGET_VAR are used as features.
#   - The script automatically converts binary-like variables to factor("0","1").
#   - H2O is initialized and shut down automatically.
# -------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(caret)
  library(h2o)
  library(reshape2)
})

# ----------------------------- 1) Config -----------------------------

SEED        <- 20241107
N_ITER      <- 20
MAX_MEM     <- "16G"      # Adjust to your available system memory
NTHREADS    <- -1         # Use all cores

# Name of the binary target variable (must exist in `input_data`)
TARGET_VAR  <- "op24"

# Optional: specify a subset of features. If NULL, use all except the target.
FEATURE_VARS <- NULL

# H2O random search hyperparameters
HYPER_PARAMS <- list(
  ntrees       = seq(50, 500, by = 50),
  max_depth    = 3:15,                 
  sample_rate  = seq(0.6, 1.0, by = 0.1)
  # mtries will be added dynamically based on number of features
)

# Random search criteria
SEARCH_CRITERIA <- list(
  strategy           = "RandomDiscrete",
  max_runtime_secs   = 3000,
  stopping_metric    = "AUCPR",
  stopping_tolerance = 0.001,
  stopping_rounds    = 3
)

# --------------------------- 2) Data Prep ---------------------------

if (!exists("input_data")) {
  stop("Please load your dataset into an object named 'input_data' before running this script.")
}

df <- input_data

# Ensure target variable exists
if (!TARGET_VAR %in% names(df)) {
  stop("Target variable '", TARGET_VAR, "' not found in the dataset.")
}

# Use all columns except target if no subset specified
if (is.null(FEATURE_VARS)) {
  FEATURE_VARS <- setdiff(names(df), TARGET_VAR)
} else {
  missing_cols <- setdiff(FEATURE_VARS, names(df))
  if (length(missing_cols) > 0) {
    stop("Missing specified feature columns: ", paste(missing_cols, collapse = ", "))
  }
}

# ---------------------- Feature Preprocessing -----------------------


# Convert binary-like columns (0/1) to factors
to_factor_if_binary <- function(v) {
  uniq <- unique(na.omit(v))
  if (all(uniq %in% c(0, 1)) || all(uniq %in% c("0", "1"))) {
    return(factor(as.character(v), levels = c("0", "1")))
  }
  v
}

for (nm in names(df)) {
  df[[nm]] <- to_factor_if_binary(df[[nm]])
}

# Ensure target is a binary factor
if (!is.factor(df[[TARGET_VAR]])) {
  df[[TARGET_VAR]] <- factor(as.character(df[[TARGET_VAR]]), levels = c("0", "1"))
}

# ---------------------- Feature selection ----------------------------

# If no feature subset is specified, use all except the target
if (is.null(FEATURE_VARS)) {
  FEATURES <- setdiff(names(df), TARGET_VAR)
} else {
  # Validate the manually specified subset
  missing_cols <- setdiff(FEATURE_VARS, names(df))
  if (length(missing_cols) > 0) {
    stop("Missing specified feature columns: ", paste(missing_cols, collapse = ", "))
  }
  FEATURES <- FEATURE_VARS
}


# --------------------- 3) Start H2O & Helpers -----------------------

`%||%` <- function(a, b) if (!is.null(a)) a else b

set.seed(SEED)
h2o.init(nthreads = NTHREADS, max_mem_size = MAX_MEM)
on.exit({
  try(h2o.shutdown(prompt = FALSE), silent = TRUE)
}, add = TRUE)

# Dynamically add mtries based on number of features
nfeat <- length(FEATURES)
hyper_params_grid <- HYPER_PARAMS
hyper_params_grid$mtries <- if (nfeat > 1) 2:nfeat else 1

train_one_rf <- function(df_local, target, features,
                         hyper_params = hyper_params_grid,
                         search_criteria = SEARCH_CRITERIA,
                         seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # Stratified 80/20 train-test split
  idx <- caret::createDataPartition(df_local[[target]], p = 0.8, list = FALSE)
  train_df <- df_local[idx, , drop = FALSE]
  test_df  <- df_local[-idx, , drop = FALSE]

  train_h2o <- as.h2o(train_df)
  test_h2o  <- as.h2o(test_df)

  # Random grid search
  grid_id <- paste0("grid_rf_", as.integer(Sys.time()))
  suppressWarnings({
    h2o.grid(
      algorithm       = "randomForest",
      grid_id         = grid_id,
      x               = features,
      y               = target,
      training_frame  = train_h2o,
      nfolds          = 10,  # restored from original code
      seed            = seed %||% 1,
      hyper_params    = hyper_params,
      search_criteria = search_criteria
    )
  })

  grid_perf <- h2o.getGrid(grid_id = grid_id, sort_by = "aucpr", decreasing = TRUE)
  if (length(grid_perf@model_ids) == 0) stop("Grid search returned no models.")
  best_id <- grid_perf@model_ids[[1]]
  best_rf <- h2o.getModel(best_id)

  perf   <- h2o.performance(best_rf, newdata = test_h2o)
  auc    <- tryCatch(h2o.auc(perf),   error = function(e) NA_real_)
  aucpr  <- tryCatch(h2o.aucpr(perf), error = function(e) NA_real_)

  metrics_tbl <- tryCatch(h2o.metric(perf), error = function(e) NULL)
  metrics_df <- if (!is.null(metrics_tbl)) {
    data.frame(precision = metrics_tbl$precision,
               recall    = metrics_tbl$recall,
               tpr       = metrics_tbl$tpr,
               fpr       = metrics_tbl$fpr)
  } else data.frame()

  var_imp <- tryCatch(h2o.varimp(best_rf), error = function(e) NULL)
  imp_df <- if (!is.null(var_imp)) {
    data.frame(variable = var_imp[, 1],
               relative_importance = var_imp[, 2],
               scaled_importance   = var_imp[, 3])
  } else data.frame()

  shap_df <- tryCatch(as.data.frame(h2o.predict_contributions(best_rf, test_h2o)),
                      error = function(e) data.frame())

  list(
    best_model_id = best_id,
    auc           = auc,
    aucpr         = aucpr,
    metrics       = metrics_df,
    importance    = imp_df,
    shap          = shap_df,
    features_test = test_df[, features, drop = FALSE]
  )
}

# --------------------- 4) Iterative Training -----------------------

auc_results     <- data.frame(iteration = integer(), auc = numeric(), aucpr = numeric())
importance_list <- vector("list", N_ITER)
metrics_list    <- vector("list", N_ITER)
shap_list       <- vector("list", N_ITER)
features_list   <- vector("list", N_ITER)
model_ids       <- character(N_ITER)

for (i in seq_len(N_ITER)) {
  cat(sprintf(">>> Iteration %d / %d\n", i, N_ITER))
  res <- train_one_rf(df_local = df, target = TARGET_VAR, features = FEATURES, seed = SEED + i)

  auc_results <- rbind(auc_results, data.frame(iteration = i, auc = res$auc, aucpr = res$aucpr))
  importance_list[[i]] <- cbind(iteration = i, res$importance)
  metrics_list[[i]]    <- cbind(iteration = i, res$metrics)
  shap_list[[i]]       <- cbind(iteration = i, res$shap)
  features_list[[i]]   <- cbind(iteration = i, res$features_test)
  model_ids[i]         <- res$best_model_id

  # Clean H2O frames between iterations (keep models)
  try(h2o.removeAll(), silent = TRUE)
}

# ---------------------- 5) Results Summary -------------------------

results <- list(
  auc_results     = auc_results,
  importance_list = importance_list,
  metrics_list    = metrics_list,
  shap_list       = shap_list,
  features_list   = features_list,
  model_ids       = model_ids
)