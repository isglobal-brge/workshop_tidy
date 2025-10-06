library(tidymodels)
library(tidyverse)
library(palmerpenguins)
library(modeldata)
library(vip)
library(corrplot)

# Set theme
theme_set(theme_minimal())

# For reproducibility
set.seed(123)

# Demonstrate bias-variance tradeoff
n <- 100
x <- seq(0, 10, length.out = n)

# True function
true_function <- function(x) sin(x) + 0.5 * x

# Generate data with noise
y <- true_function(x) + rnorm(n, sd = 0.5)
data_sim <- tibble(x = x, y = y, y_true = true_function(x))

# Fit models of different complexity
models <- list(
  "Underfit (High Bias)" = lm(y ~ x, data = data_sim),
  "Good Fit" = lm(y ~ poly(x, 3), data = data_sim),
  "Overfit (High Variance)" = lm(y ~ poly(x, 15), data = data_sim)
)

# Generate predictions
predictions <- map_df(names(models), function(model_name) {
  model <- models[[model_name]]
  tibble(
    x = x,
    y_true = true_function(x),
    y_pred = predict(model),
    model = model_name
  )
})

# Visualize
ggplot() +
  geom_point(data = data_sim, aes(x = x, y = y), alpha = 0.3) +
  geom_line(data = data_sim, aes(x = x, y = y_true), 
            color = "black", linewidth = 1.5, linetype = "dashed") +
  geom_line(data = predictions, aes(x = x, y = y_pred, color = model), 
            linewidth = 1.2) +
  facet_wrap(~model, ncol = 3) +
  scale_color_manual(values = c("red", "green", "blue")) +
  labs(
    title = "Bias-Variance Tradeoff Demonstration",
    subtitle = "Black dashed line = true function, Points = observed data",
    x = "X",
    y = "Y"
  ) +
  theme(legend.position = "none")

# Create training and test sets
set.seed(123)
train_indices <- sample(1:n, size = 0.7 * n)
train_data <- data_sim[train_indices, ]
test_data <- data_sim[-train_indices, ]

# Fit models of increasing complexity
complexity_range <- 1:12
train_errors <- numeric(length(complexity_range))
test_errors <- numeric(length(complexity_range))

for (i in complexity_range) {
  model <- lm(y ~ poly(x, i), data = train_data)
  
  train_pred <- predict(model, train_data)
  test_pred <- predict(model, test_data)
  
  train_errors[i] <- mean((train_data$y - train_pred)^2)
  test_errors[i] <- mean((test_data$y - test_pred)^2)
}

# Plot training vs test error
error_data <- tibble(
  complexity = rep(complexity_range, 2),
  error = c(train_errors, test_errors),
  type = rep(c("Training", "Test"), each = length(complexity_range))
)

ggplot(error_data, aes(x = complexity, y = error, color = type)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
  labs(
    title = "Training vs Test Error: Detecting Overfitting",
    subtitle = "Test error increases while training error decreases = Overfitting",
    x = "Model Complexity (Polynomial Degree)",
    y = "Mean Squared Error",
    color = "Dataset"
  ) +
  geom_vline(xintercept = 3, linetype = "dashed", alpha = 0.5) +
  annotate("text", x = 3, y = max(test_errors) * 0.9, 
           label = "Optimal\nComplexity", hjust = -0.1)

# Core tidymodels packages
tidymodels_packages <- c(
  "rsample",     # Data splitting and resampling
  "parsnip",     # Model specification
  "recipes",     # Feature engineering
  "workflows",   # Workflow management
  "tune",        # Hyperparameter tuning
  "yardstick",   # Model evaluation metrics
  "broom",       # Tidy model outputs
  "dials"        # Parameter tuning dials
)

# Display package info
tibble(
  Package = tidymodels_packages,
  Purpose = c(
    "Data splitting, cross-validation, bootstrapping",
    "Unified interface for model specification",
    "Feature engineering and preprocessing",
    "Combine preprocessing and modeling",
    "Hyperparameter optimization",
    "Performance metrics and evaluation",
    "Convert model outputs to tidy format",
    "Tools for creating tuning parameter sets"
  )
) %>%
  knitr::kable()

# Load and explore data
penguins_clean <- penguins %>%
  drop_na()

# Basic exploration
glimpse(penguins_clean)

# Class distribution
penguins_clean %>%
  count(species) %>%
  mutate(prop = n / sum(n))

# Correlation matrix
penguins_clean %>%
  select(where(is.numeric)) %>%
  cor() %>%
  corrplot(method = "color", type = "upper", 
           order = "hclust", tl.cex = 0.8,
           addCoef.col = "black", number.cex = 0.7)

# Feature relationships
penguins_clean %>%
  select(species, bill_length_mm, bill_depth_mm, 
         flipper_length_mm, body_mass_g) %>%
  pivot_longer(cols = -species, names_to = "measurement", values_to = "value") %>%
  ggplot(aes(x = value, fill = species)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~measurement, scales = "free") +
  scale_fill_viridis_d() +
  labs(title = "Feature Distributions by Species")

# Initial split (training vs testing)
set.seed(123)
penguin_split <- initial_split(penguins_clean, prop = 0.75, strata = species)

penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)

# Check split proportions
tibble(
  Dataset = c("Training", "Testing"),
  N = c(nrow(penguin_train), nrow(penguin_test)),
  Proportion = c(nrow(penguin_train), nrow(penguin_test)) / nrow(penguins_clean)
) %>%
  knitr::kable()

# Create cross-validation folds
penguin_folds <- vfold_cv(penguin_train, v = 5, strata = species)
penguin_folds

# Create a recipe
penguin_recipe <- recipe(species ~ ., data = penguin_train) %>%
  # Remove unnecessary variables
  step_rm(year) %>%
  # Convert factors to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  # Normalize numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero variance predictors
  step_zv(all_predictors())

# View the recipe
penguin_recipe

# Prepare and bake to see transformed data
penguin_prep <- prep(penguin_recipe)
bake(penguin_prep, new_data = penguin_train %>% head())

# Specify different models

# Multinomial regression (for multiclass classification)
multinom_spec <- multinom_reg() %>%
  set_engine("nnet") %>%
  set_mode("classification")

# Random forest
rf_spec <- rand_forest(
  trees = 100,
  min_n = 5
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# Support vector machine
svm_spec <- svm_rbf(
  cost = 1,
  rbf_sigma = 0.01
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

print("Model specifications created")

# Combine recipe and model into workflows
multinom_workflow <- workflow() %>%
  add_recipe(penguin_recipe) %>%
  add_model(multinom_spec)

rf_workflow <- workflow() %>%
  add_recipe(penguin_recipe) %>%
  add_model(rf_spec)

svm_workflow <- workflow() %>%
  add_recipe(penguin_recipe) %>%
  add_model(svm_spec)

multinom_workflow

# Fit models using cross-validation
# For multiclass problems, we'll use accuracy and multiclass AUC
multinom_cv <- fit_resamples(
  multinom_workflow,
  resamples = penguin_folds,
  metrics = metric_set(accuracy, roc_auc),
  control = control_resamples(save_pred = TRUE)
)

rf_cv <- fit_resamples(
  rf_workflow,
  resamples = penguin_folds,
  metrics = metric_set(accuracy, roc_auc),
  control = control_resamples(save_pred = TRUE)
)

# Compare models
model_comparison <- bind_rows(
  collect_metrics(multinom_cv) %>% mutate(model = "Multinomial Regression"),
  collect_metrics(rf_cv) %>% mutate(model = "Random Forest")
)

# Visualize comparison
ggplot(model_comparison, aes(x = model, y = mean, fill = model)) +
  geom_col() +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  scale_fill_viridis_d() +
  labs(
    title = "Model Performance Comparison",
    subtitle = "5-fold cross-validation results",
    y = "Score"
  ) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

# Train final model on full training set
final_model <- rf_workflow %>%
  fit(penguin_train)

# Make predictions on test set
predictions <- final_model %>%
  predict(penguin_test) %>%
  bind_cols(penguin_test %>% select(species))

# Confusion matrix
conf_mat <- predictions %>%
  conf_mat(truth = species, estimate = .pred_class)

conf_mat

# Visualize confusion matrix
autoplot(conf_mat, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "darkblue") +
  labs(title = "Confusion Matrix - Random Forest")

# Feature importance
final_rf <- final_model %>%
  extract_fit_parsnip()

# Variable importance plot
if (require(vip, quietly = TRUE)) {
  vip(final_rf, num_features = 10) +
    labs(title = "Feature Importance - Random Forest")
}

# Prediction probabilities
prob_predictions <- final_model %>%
  predict(penguin_test, type = "prob") %>%
  bind_cols(penguin_test %>% select(species))

# ROC curves for multiclass
if (require(yardstick, quietly = TRUE)) {
  roc_data <- prob_predictions %>%
    roc_curve(truth = species, .pred_Adelie:.pred_Gentoo)
  
  autoplot(roc_data) +
    labs(
      title = "ROC Curves by Species",
      subtitle = "One-vs-All approach"
    )
}

# Demonstrate different CV strategies
set.seed(123)
sample_data <- tibble(
  id = 1:100,
  x = rnorm(100),
  y = 2 * x + rnorm(100, sd = 0.5),
  group = rep(1:10, each = 10),
  time = rep(1:10, 10)
)

# Different CV strategies
cv_strategies <- list(
  "5-Fold CV" = vfold_cv(sample_data, v = 5),
  "10-Fold CV" = vfold_cv(sample_data, v = 10),
  "Leave-One-Out CV" = loo_cv(sample_data),
  "Bootstrap" = bootstraps(sample_data, times = 5),
  "Group CV" = group_vfold_cv(sample_data, group = group, v = 5)
)

# Visualize fold assignments
fold_viz <- vfold_cv(sample_data, v = 5) %>%
  mutate(fold_data = map(splits, analysis)) %>%
  unnest(fold_data, names_sep = "_") %>%
  select(obs_id = fold_data_id, Fold = id) %>%
  distinct()

ggplot(fold_viz, aes(x = obs_id, y = 1, fill = Fold)) +
  geom_tile(height = 0.8) +
  scale_fill_viridis_d() +
  labs(
    title = "5-Fold Cross-Validation: Data Assignment",
    subtitle = "Each observation appears in exactly one test fold",
    x = "Observation ID",
    y = ""
  ) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

# Demonstrate AIC/BIC for model selection
models_to_compare <- list(
  "Simple" = lm(body_mass_g ~ bill_length_mm, data = penguin_train),
  "Moderate" = lm(body_mass_g ~ bill_length_mm + flipper_length_mm, data = penguin_train),
  "Complex" = lm(body_mass_g ~ bill_length_mm + flipper_length_mm + 
                 bill_depth_mm + island + sex, data = penguin_train),
  "Very Complex" = lm(body_mass_g ~ .^2, data = penguin_train)  # All interactions
)

model_selection <- map_df(names(models_to_compare), function(name) {
  model <- models_to_compare[[name]]
  tibble(
    Model = name,
    Parameters = length(coef(model)),
    AIC = AIC(model),
    BIC = BIC(model),
    `Adj R²` = summary(model)$adj.r.squared
  )
})

model_selection %>%
  arrange(AIC) %>%
  knitr::kable(digits = 2)

# Demonstrate regularization
library(glmnet)

# Prepare data
X <- model.matrix(body_mass_g ~ . - 1, data = penguin_train %>% select(-species, -year))
y <- penguin_train$body_mass_g

# Fit ridge and lasso
ridge_fit <- glmnet(X, y, alpha = 0)  # Ridge
lasso_fit <- glmnet(X, y, alpha = 1)  # Lasso

# Plot coefficient paths
par(mfrow = c(1, 2))
plot(ridge_fit, xvar = "lambda", main = "Ridge Regression")
plot(lasso_fit, xvar = "lambda", main = "Lasso Regression")

# WRONG: Preprocessing before splitting
# This leaks information from test set into training
wrong_way <- penguins_clean %>%
  mutate(bill_length_scaled = scale(bill_length_mm)[,1])  # Uses all data!

# RIGHT: Preprocessing within training set only
right_way <- recipe(species ~ ., data = penguin_train) %>%
  step_normalize(all_numeric_predictors())  # Only uses training data

# Your solution
cv_comparison <- tibble(
  strategy = c("5-Fold", "10-Fold", "Bootstrap", "Monte Carlo"),
  cv_object = list(
    vfold_cv(penguin_train, v = 5),
    vfold_cv(penguin_train, v = 10),
    bootstraps(penguin_train, times = 25),
    mc_cv(penguin_train, prop = 0.75, times = 25)
  )
)

# Fit a simple model with each CV strategy
simple_spec <- multinom_reg() %>%
  set_engine("nnet")

simple_recipe <- recipe(species ~ bill_length_mm + bill_depth_mm, 
                       data = penguin_train) %>%
  step_normalize(all_predictors())

simple_workflow <- workflow() %>%
  add_recipe(simple_recipe) %>%
  add_model(simple_spec)

# Compare results
cv_results <- cv_comparison %>%
  mutate(
    fits = map(cv_object, ~ fit_resamples(simple_workflow, resamples = .)),
    metrics = map(fits, collect_metrics)
  ) %>%
  unnest(metrics) %>%
  filter(.metric == "accuracy") %>%
  select(strategy, mean, std_err)

cv_results %>%
  ggplot(aes(x = strategy, y = mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), width = 0.2) +
  labs(title = "Cross-Validation Strategy Comparison",
       y = "Accuracy")

# Your solution
# Create polynomial features of different degrees
complexity_levels <- 1:5

model_fits <- map(complexity_levels, function(degree) {
  recipe_poly <- recipe(body_mass_g ~ flipper_length_mm, data = penguin_train) %>%
    step_poly(flipper_length_mm, degree = degree)
  
  lm_spec <- linear_reg() %>%
    set_engine("lm")
  
  workflow() %>%
    add_recipe(recipe_poly) %>%
    add_model(lm_spec) %>%
    fit(penguin_train)
})

# Evaluate on training and test sets
evaluation <- map_df(1:length(model_fits), function(i) {
  model <- model_fits[[i]]
  
  train_pred <- predict(model, penguin_train)
  test_pred <- predict(model, penguin_test)
  
  tibble(
    complexity = complexity_levels[i],
    train_rmse = rmse_vec(penguin_train$body_mass_g, train_pred$.pred),
    test_rmse = rmse_vec(penguin_test$body_mass_g, test_pred$.pred)
  )
})

evaluation %>%
  pivot_longer(cols = c(train_rmse, test_rmse), 
               names_to = "dataset", values_to = "rmse") %>%
  ggplot(aes(x = complexity, y = rmse, color = dataset)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  labs(title = "Model Complexity vs Error",
       x = "Polynomial Degree",
       y = "RMSE") +
  scale_color_manual(values = c("train_rmse" = "blue", "test_rmse" = "red"))

# Your solution
# Predict penguin body mass
mass_split <- initial_split(penguins_clean, prop = 0.8)
mass_train <- training(mass_split)
mass_test <- testing(mass_split)

# Create recipe with feature engineering
mass_recipe <- recipe(body_mass_g ~ ., data = mass_train) %>%
  step_rm(year) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_interact(terms = ~ bill_length_mm:bill_depth_mm)

# Specify model
rf_reg_spec <- rand_forest(
  trees = 200,
  min_n = 10
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

# Create workflow
mass_workflow <- workflow() %>%
  add_recipe(mass_recipe) %>%
  add_model(rf_reg_spec)

# Fit and evaluate
mass_fit <- mass_workflow %>%
  fit(mass_train)

# Predictions
mass_predictions <- mass_fit %>%
  predict(mass_test) %>%
  bind_cols(mass_test)

# Evaluate
metrics <- mass_predictions %>%
  metrics(truth = body_mass_g, estimate = .pred)

print(metrics)

# Visualize predictions
ggplot(mass_predictions, aes(x = body_mass_g, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Body Mass",
       x = "Actual Mass (g)",
       y = "Predicted Mass (g)")
