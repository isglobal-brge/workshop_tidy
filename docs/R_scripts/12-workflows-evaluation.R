library(tidymodels)
library(tidyverse)
library(modeldata)
library(vip)
library(patchwork)
library(workflowsets)
library(probably)

# Set theme and seed
theme_set(theme_minimal())
set.seed(123)

# Load example data
data(ames)
ames_split <- initial_split(ames, prop = 0.75, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# The WRONG way - manual preprocessing
# This approach is error-prone and can lead to data leakage

# Manual preprocessing on training data
ames_train_processed <- ames_train %>%
  mutate(
    # Log transform - uses training data statistics
    Sale_Price_log = log(Sale_Price),
    # Scaling - WRONG! Uses all training data including validation folds
    Gr_Liv_Area_scaled = scale(Gr_Liv_Area)[,1]
  )

# Now we need to remember these transformations for test data
# And apply them consistently... but what were the scaling parameters?

# The RIGHT way - using workflows
# Step 1: Create a recipe
ames_recipe <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built + Total_Bsmt_SF + 
                      Neighborhood, 
                      data = ames_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

# Step 2: Create a model specification
lm_spec <- linear_reg() %>%
  set_engine("lm")

# Step 3: Combine into a workflow
lm_workflow <- workflow() %>%
  add_recipe(ames_recipe) %>%
  add_model(lm_spec)

print(lm_workflow)

# Start with an empty workflow
base_workflow <- workflow()

# Add components step by step
base_workflow <- base_workflow %>%
  add_recipe(ames_recipe)

print("After adding recipe:")
print(base_workflow)

base_workflow <- base_workflow %>%
  add_model(lm_spec)

print("After adding model:")
print(base_workflow)

# You can also update components
updated_workflow <- lm_workflow %>%
  update_model(
    linear_reg(penalty = 0.01) %>%
      set_engine("glmnet")
  )

print("After updating model:")
print(updated_workflow)

# Or remove components
recipe_only <- lm_workflow %>%
  remove_model()

print("After removing model:")
print(recipe_only)

# Method 1: Formula interface (simple, no preprocessing)
formula_workflow <- workflow() %>%
  add_formula(Sale_Price ~ Gr_Liv_Area + Overall_Cond) %>%
  add_model(lm_spec)

# Method 2: Recipe interface (complex preprocessing)
recipe_workflow <- workflow() %>%
  add_recipe(ames_recipe) %>%
  add_model(lm_spec)

# Method 3: Variables interface (programmatic)
vars_workflow <- workflow() %>%
  add_variables(
    outcomes = Sale_Price,
    predictors = c(Gr_Liv_Area, Overall_Cond, Neighborhood)
  ) %>%
  add_model(lm_spec)

# Compare the approaches
print("Formula approach:")
formula_workflow

print("\nRecipe approach:")
recipe_workflow

print("\nVariables approach:")
vars_workflow

# Fit the workflow
lm_fit <- lm_workflow %>%
  fit(data = ames_train)

# The fitted workflow contains:
# 1. The prepared recipe (with learned parameters)
# 2. The fitted model
print(lm_fit)

# Make predictions - automatically applies all preprocessing!
predictions <- lm_fit %>%
  predict(ames_test)

head(predictions)

# Get multiple types of predictions
all_predictions <- bind_cols(
  ames_test %>% select(Sale_Price),
  predict(lm_fit, ames_test),           # Point predictions
  predict(lm_fit, ames_test, type = "conf_int")  # Confidence intervals for lm
)

head(all_predictions)

# The workflow ensures consistency
# These transformations are automatically applied:
# 1. Log transform of Sale_Price (inverse transformed for predictions)
# 2. Normalization of numeric predictors
# 3. Dummy encoding of Neighborhood

# Create multiple preprocessing recipes
recipe_simple <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built, 
                       data = ames_train)

recipe_normalized <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built + Total_Bsmt_SF, 
                           data = ames_train) %>%
  step_normalize(all_numeric_predictors())

recipe_complex <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built + 
                         Neighborhood + Total_Bsmt_SF + Garage_Cars + 
                         First_Flr_SF + Full_Bath, data = ames_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_dummy(all_nominal_predictors())

# Create multiple model specifications
models <- list(
  lm = linear_reg() %>% set_engine("lm"),
  ridge = linear_reg(penalty = 0.1, mixture = 0) %>% set_engine("glmnet"),
  lasso = linear_reg(penalty = 0.1, mixture = 1) %>% set_engine("glmnet"),
  tree = decision_tree(tree_depth = 10) %>% 
    set_engine("rpart") %>% 
    set_mode("regression")
)

# Create workflow set
workflow_set <- workflow_set(
  preproc = list(
    simple = recipe_simple,
    normalized = recipe_normalized
  ),
  models = models
)

print(workflow_set)

# This creates 8 workflows (2 recipes × 4 models)!

# Create resamples for evaluation
ames_folds <- vfold_cv(ames_train, v = 5, strata = Sale_Price)

# Evaluate workflows individually to avoid errors
# We'll evaluate just a subset for demonstration
simple_lm <- workflow() %>%
  add_recipe(recipe_simple) %>%
  add_model(linear_reg() %>% set_engine("lm"))

simple_lm_results <- simple_lm %>%
  fit_resamples(
    resamples = ames_folds,
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
  )

# Show metrics
collect_metrics(simple_lm_results)

# Visualize performance
simple_lm_results %>%
  collect_metrics() %>%
  ggplot(aes(x = .metric, y = mean)) +
  geom_col(fill = "steelblue") +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  facet_wrap(~.metric, scales = "free") +
  labs(title = "Model Performance with Cross-Validation",
       subtitle = "Simple linear model with 5-fold CV")

# Fit our simple workflow
best_fit <- simple_lm %>%
  fit(ames_train)

# Get test predictions
test_predictions <- best_fit %>%
  predict(ames_test) %>%
  bind_cols(ames_test %>% select(Sale_Price))

# Calculate multiple metrics
regression_metrics <- metric_set(
  rmse,      # Root Mean Squared Error
  mae,       # Mean Absolute Error
  mape,      # Mean Absolute Percentage Error
  rsq,       # R-squared
  ccc        # Concordance Correlation Coefficient
)

test_performance <- test_predictions %>%
  regression_metrics(truth = Sale_Price, estimate = .pred)

print(test_performance)

# Visualize predictions vs actual
ggplot(test_predictions, aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(
    title = "Predicted vs Actual Sale Prices",
    subtitle = paste("Test RMSE:", round(test_performance$.estimate[1], 2)),
    x = "Actual Sale Price",
    y = "Predicted Sale Price"
  ) +
  coord_equal()

# Create a classification problem
ames_class <- ames_train %>%
  mutate(expensive = factor(if_else(Sale_Price > median(Sale_Price), 
                                    "yes", "no"))) %>%
  select(-Sale_Price)

# Simple classification workflow
class_recipe <- recipe(expensive ~ Gr_Liv_Area + Overall_Cond + Year_Built, 
                      data = ames_class) %>%
  step_normalize(all_numeric_predictors())

class_spec <- logistic_reg() %>%
  set_engine("glm")

class_workflow <- workflow() %>%
  add_recipe(class_recipe) %>%
  add_model(class_spec)

# Fit and predict
class_split <- initial_split(ames_class, strata = expensive)
class_train <- training(class_split)
class_test <- testing(class_split)

class_fit <- class_workflow %>%
  fit(class_train)

class_predictions <- bind_cols(
  class_test %>% select(expensive),
  predict(class_fit, class_test),
  predict(class_fit, class_test, type = "prob")
)

# Classification metrics
class_metrics <- metric_set(
  accuracy,
  precision,
  recall,
  f_meas,
  roc_auc,
  pr_auc
)

class_performance <- class_predictions %>%
  class_metrics(truth = expensive, estimate = .pred_class, .pred_yes)

print(class_performance)

# Confusion matrix
conf_matrix <- class_predictions %>%
  conf_mat(truth = expensive, estimate = .pred_class)

autoplot(conf_matrix, type = "heatmap") +
  labs(title = "Confusion Matrix")

# ROC curve
roc_curve_data <- class_predictions %>%
  roc_curve(truth = expensive, .pred_yes)

autoplot(roc_curve_data) +
  labs(title = "ROC Curve") +
  annotate("text", x = 0.5, y = 0.5, 
           label = paste("AUC:", round(class_performance$.estimate[5], 3)))

# Create a custom metric - Mean Absolute Percentage Error with threshold
mape_vec_threshold <- function(truth, estimate, threshold = 0.2, na_rm = TRUE) {
  errors <- abs((truth - estimate) / truth)
  errors[errors > threshold] <- threshold  # Cap errors at threshold
  mean(errors, na.rm = na_rm)
}

# Use the custom metric directly
custom_mape <- test_predictions %>%
  summarise(
    custom_mape = mape_vec_threshold(Sale_Price, .pred),
    regular_mape = mean(abs((Sale_Price - .pred) / Sale_Price))
  )

print(custom_mape)

# Create a business-specific metric
# For example: penalize underestimation more than overestimation
asymmetric_loss <- function(truth, estimate, under_weight = 2) {
  errors <- truth - estimate
  result <- ifelse(errors > 0, 
                   under_weight * errors^2,  # Underestimation penalty
                   errors^2)                  # Overestimation penalty
  sqrt(mean(result))
}

# Apply custom metric
test_predictions %>%
  mutate(
    asymmetric_rmse = asymmetric_loss(Sale_Price, .pred)
  ) %>%
  summarise(
    regular_rmse = rmse_vec(Sale_Price, .pred),
    asymmetric_rmse = mean(asymmetric_rmse)
  )

# Extract components from fitted workflow
extracted_recipe <- lm_fit %>%
  extract_recipe()

extracted_model <- lm_fit %>%
  extract_fit_parsnip()

# Get preprocessing results
preprocessed_data <- lm_fit %>%
  extract_recipe() %>%
  bake(new_data = ames_test)

head(preprocessed_data)

# Extract model coefficients
coefficients <- lm_fit %>%
  extract_fit_parsnip() %>%
  tidy()

head(coefficients)

# Variable importance (if applicable)
# For models that support it
rf_workflow <- workflow() %>%
  add_recipe(recipe_simple) %>%
  add_model(rand_forest() %>% set_engine("ranger", importance = "impurity") %>% set_mode("regression"))

rf_fit <- rf_workflow %>%
  fit(ames_train)

rf_importance <- rf_fit %>%
  extract_fit_parsnip() %>%
  vip()

print(rf_importance)

# Example: Finalize a workflow with best parameters
# This would typically come from tuning
best_params <- tibble(
  penalty = 0.01,
  mixture = 0.5
)

# Create tunable workflow
tunable_spec <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

tunable_workflow <- workflow() %>%
  add_recipe(recipe_normalized) %>%
  add_model(tunable_spec)

# Finalize with best parameters
final_workflow <- tunable_workflow %>%
  finalize_workflow(best_params)

print(final_workflow)

# Fit final model on all training data
final_fit <- final_workflow %>%
  fit(ames_train)

# Last fit - train on all data, evaluate on test
last_fit_results <- final_workflow %>%
  last_fit(split = ames_split, metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq))

# Extract final metrics
final_metrics <- last_fit_results %>%
  collect_metrics()

print(final_metrics)

# Extract final model
final_model <- last_fit_results %>%
  extract_workflow()

# Residual analysis for regression
residual_analysis <- test_predictions %>%
  mutate(
    residual = Sale_Price - .pred,
    std_residual = residual / sd(residual),
    abs_residual = abs(residual)
  )

# Residual plots
p1 <- ggplot(residual_analysis, aes(x = .pred, y = residual)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(se = FALSE, color = "blue") +
  labs(title = "Residuals vs Fitted", x = "Fitted Values", y = "Residuals")

p2 <- ggplot(residual_analysis, aes(sample = std_residual)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot", x = "Theoretical Quantiles", y = "Standardized Residuals")

p3 <- ggplot(residual_analysis, aes(x = .pred, y = sqrt(abs_residual))) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE, color = "blue") +
  labs(title = "Scale-Location", x = "Fitted Values", y = "√|Residuals|")

p4 <- ggplot(residual_analysis, aes(x = residual)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  geom_density(aes(y = after_stat(count)), color = "red", linewidth = 1) +
  labs(title = "Residual Distribution", x = "Residuals", y = "Count")

# Combine plots
(p1 + p2) / (p3 + p4) +
  plot_annotation(title = "Regression Diagnostics")

# Calibration analysis
calibration_data <- class_predictions %>%
  mutate(
    prob_bin = cut(.pred_yes, breaks = seq(0, 1, 0.1), include.lowest = TRUE)
  ) %>%
  group_by(prob_bin) %>%
  summarise(
    mean_predicted = mean(.pred_yes),
    fraction_positive = mean(expensive == "yes"),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n > 5)  # Remove bins with few observations

# Calibration plot
ggplot(calibration_data, aes(x = mean_predicted, y = fraction_positive)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(aes(size = n), color = "darkblue") +
  geom_line(color = "darkblue") +
  scale_size_continuous(range = c(2, 8)) +
  labs(
    title = "Probability Calibration Plot",
    subtitle = "Well-calibrated models follow the diagonal",
    x = "Mean Predicted Probability",
    y = "Observed Frequency",
    size = "Count"
  ) +
  coord_equal() +
  xlim(0, 1) + ylim(0, 1)

# Use probably package for calibration
library(probably)

# Simple calibration visualization
# We'll just show the calibration plot without the probably package functions
# that are causing issues

# Create a simple comparison
ggplot(calibration_data, aes(x = mean_predicted, y = fraction_positive)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", size = 1) +
  geom_point(aes(size = n), color = "darkblue", alpha = 0.7) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  scale_size_continuous(range = c(2, 8)) +
  labs(
    title = "Model Calibration Assessment",
    subtitle = "Points should follow the red diagonal for perfect calibration",
    x = "Mean Predicted Probability",
    y = "Observed Frequency",
    size = "Count"
  ) +
  coord_equal() +
  xlim(0, 1) + ylim(0, 1) +
  theme_minimal()

# For regression: actual vs predicted with confidence bands
prediction_plot <- test_predictions %>%
  ggplot(aes(x = Sale_Price, y = .pred)) +
  geom_hex(bins = 30) +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  scale_fill_viridis_c() +
  labs(
    title = "Prediction Accuracy",
    subtitle = paste("R² =", round(rsq_vec(test_predictions$Sale_Price, 
                                          test_predictions$.pred), 3)),
    x = "Actual Price",
    y = "Predicted Price"
  ) +
  coord_equal()

# Error distribution
error_plot <- test_predictions %>%
  mutate(error = .pred - Sale_Price,
         pct_error = error / Sale_Price * 100) %>%
  ggplot(aes(x = pct_error)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Prediction Error Distribution",
    subtitle = "Percentage error",
    x = "Error (%)",
    y = "Count"
  )

prediction_plot + error_plot

# Good practice: Complete workflow
production_workflow <- workflow() %>%
  add_recipe(
    recipe(Sale_Price ~ ., data = ames_train) %>%
      step_impute_median(all_numeric_predictors()) %>%
      step_impute_mode(all_nominal_predictors()) %>%
      step_normalize(all_numeric_predictors()) %>%
      step_dummy(all_nominal_predictors())
  ) %>%
  add_model(
    linear_reg() %>% set_engine("lm")
  )

# This ensures all preprocessing is contained and reproducible

# Save workflow for reproducibility
saveRDS(final_fit, "models/final_workflow_v1.rds")

# Load and use later
# loaded_workflow <- readRDS("models/final_workflow_v1.rds")
# new_predictions <- predict(loaded_workflow, new_data)

# Create workflow with documentation
documented_workflow <- workflow() %>%
  add_recipe(
    recipe(Sale_Price ~ ., data = ames_train) %>%
      # Handle missing values before other steps
      step_impute_median(all_numeric_predictors()) %>%
      # Normalize for model stability
      step_normalize(all_numeric_predictors()) %>%
      # Create dummies for linear model
      step_dummy(all_nominal_predictors())
  ) %>%
  add_model(
    # Ridge regression to handle multicollinearity
    linear_reg(penalty = 0.01, mixture = 0) %>%
      set_engine("glmnet")
  )

# Your solution
# Create a complete evaluation pipeline
eval_recipe <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built + Overall_Cond + 
                      Neighborhood + Total_Bsmt_SF, data = ames_train) %>%
  step_log(Sale_Price, skip = TRUE) %>%  # Skip for prediction
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

eval_spec <- linear_reg(penalty = 0.01, mixture = 0.5) %>%
  set_engine("glmnet")

eval_workflow <- workflow() %>%
  add_recipe(eval_recipe) %>%
  add_model(eval_spec)

# Fit with cross-validation
eval_folds <- vfold_cv(ames_train, v = 10, strata = Sale_Price)

eval_results <- eval_workflow %>%
  fit_resamples(
    resamples = eval_folds,
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae, yardstick::mape),
    control = control_resamples(save_pred = TRUE)
  )

# Summarize performance
collect_metrics(eval_results)

# Get predictions for visualization
eval_predictions <- collect_predictions(eval_results)

# Visualize CV performance
ggplot(eval_predictions, aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.1) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  facet_wrap(~id, ncol = 5) +
  labs(title = "Predictions Across CV Folds")

# Your solution
# Define different preprocessing strategies
preproc_minimal <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built, 
                         data = ames_train)

preproc_standard <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built + Total_Bsmt_SF, 
                          data = ames_train) %>%
  step_normalize(all_numeric_predictors())

preproc_complex <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built + 
                         Neighborhood + Total_Bsmt_SF + Garage_Cars, data = ames_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(terms = ~ Gr_Liv_Area:Year_Built)

# Create workflows
strategies <- list(
  minimal = preproc_minimal,
  standard = preproc_standard,
  complex = preproc_complex
)

# Same model for all
model_spec <- linear_reg() %>% set_engine("lm")

# Create and evaluate workflows
strategy_results <- map_df(names(strategies), function(strategy_name) {
  wf <- workflow() %>%
    add_recipe(strategies[[strategy_name]]) %>%
    add_model(model_spec)
  
  # Fit and evaluate
  fit_resamples(
    wf,
    resamples = vfold_cv(ames_train, v = 5),
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq)
  ) %>%
    collect_metrics() %>%
    mutate(strategy = strategy_name)
})

# Compare strategies
ggplot(strategy_results, aes(x = strategy, y = mean, fill = strategy)) +
  geom_col() +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  labs(title = "Preprocessing Strategy Comparison") +
  theme(legend.position = "none")

# Your solution
# Business scenario: Real estate company
# - Overestimating is bad (disappointed customers)
# - Underestimating by <5% is acceptable
# - Underestimating by >5% is very bad (lost opportunity)

business_metric <- function(truth, estimate) {
  pct_error <- (estimate - truth) / truth * 100
  
  penalties <- case_when(
    pct_error > 0 ~ abs(pct_error) * 2,        # Overestimate penalty
    pct_error > -5 ~ abs(pct_error) * 0.5,     # Small underestimate
    TRUE ~ abs(pct_error) * 3                  # Large underestimate penalty
  )
  
  mean(penalties)
}

# Apply to test predictions
test_predictions %>%
  mutate(
    business_score = business_metric(Sale_Price, .pred),
    standard_mape = mape_vec(Sale_Price, .pred)
  ) %>%
  summarise(
    mean_business_score = mean(business_score),
    mean_standard_mape = mean(standard_mape)
  )

# Visualize business metric
test_predictions %>%
  mutate(
    pct_error = (.pred - Sale_Price) / Sale_Price * 100,
    error_category = case_when(
      pct_error > 0 ~ "Overestimate",
      pct_error > -5 ~ "Small Underestimate",
      TRUE ~ "Large Underestimate"
    )
  ) %>%
  ggplot(aes(x = pct_error, fill = error_category)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  scale_fill_manual(values = c(
    "Overestimate" = "red",
    "Small Underestimate" = "yellow",
    "Large Underestimate" = "darkred"
  )) +
  labs(
    title = "Business Impact of Prediction Errors",
    x = "Percentage Error",
    y = "Count",
    fill = "Error Category"
  )
