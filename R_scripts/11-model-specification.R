library(tidymodels)
library(tidyverse)
library(modeldata)
library(ranger)     # For random forests
library(glmnet)     # For regularized regression
library(kknn)       # For k-nearest neighbors
library(kernlab)    # For support vector machines
library(xgboost)    # For gradient boosting

# Set theme and seed
theme_set(theme_minimal())
set.seed(123)

# Load data
data(ames)
ames_small <- ames %>%
  select(Sale_Price, Gr_Liv_Area, Year_Built, Overall_Cond, Neighborhood) %>%
  slice_sample(n = 500)  # Smaller sample for demonstration

# The chaos of different interfaces
# Linear regression with lm()
lm_fit <- lm(Sale_Price ~ ., data = ames_small)

# Random forest with ranger()
rf_fit <- ranger(Sale_Price ~ ., data = ames_small, num.trees = 100)

# Elastic net with glmnet() - requires matrix input!
x_matrix <- model.matrix(Sale_Price ~ . - 1, data = ames_small)
y_vector <- ames_small$Sale_Price
glmnet_fit <- glmnet(x_matrix, y_vector, alpha = 0.5)

# Different prediction methods
lm_pred <- predict(lm_fit, ames_small)  # Returns vector
rf_pred <- predict(rf_fit, ames_small)$predictions  # Returns list with $predictions
glmnet_pred <- predict(glmnet_fit, x_matrix, s = 0.01)  # Requires matrix and lambda

# The outputs are all different!
str(lm_pred)
str(rf_pred)
str(glmnet_pred)

# Step 1: Specify the model type
linear_spec <- linear_reg()
print(linear_spec)

# Step 2: Set the engine (implementation)
linear_spec <- linear_spec %>%
  set_engine("lm")
print(linear_spec)

# Step 3: Set the mode (if needed - regression vs classification)
# For linear_reg, mode is always regression, so this is automatic

# Step 4: Fit the model
linear_fit <- linear_spec %>%
  fit(Sale_Price ~ ., data = ames_small)

# Consistent prediction interface
linear_pred <- predict(linear_fit, ames_small)
str(linear_pred)  # Always returns a tibble with .pred column

# Linear regression
linear_reg_spec <- linear_reg() %>%
  set_engine("lm")

# Decision tree
tree_reg_spec <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("regression")

# Random forest
rf_reg_spec <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Support vector machine
svm_reg_spec <- svm_rbf() %>%
  set_engine("kernlab") %>%
  set_mode("regression")

# Neural network
nn_reg_spec <- mlp() %>%
  set_engine("nnet") %>%
  set_mode("regression")

# Show info for linear regression
parsnip::show_model_info("linear_reg")

# Create classification data
ames_class <- ames %>%
  mutate(expensive = factor(if_else(Sale_Price > median(Sale_Price), 
                                    "yes", "no"))) %>%
  select(expensive, Gr_Liv_Area, Year_Built, Overall_Cond) %>%
  slice_sample(n = 500)

# Logistic regression
logistic_spec <- logistic_reg() %>%
  set_engine("glm")

# Random forest for classification
rf_class_spec <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Support vector machine for classification
svm_class_spec <- svm_rbf() %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# K-nearest neighbors
knn_spec <- nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Naive Bayes
nb_spec <- naive_Bayes() %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

# See all engines for a model type
show_engines("rand_forest")

# Different engines for linear regression
linear_engines <- show_engines("linear_reg")
print(linear_engines)

# Let's compare different engines for the same model
engines_to_compare <- c("lm", "glm", "glmnet")

linear_comparisons <- map(engines_to_compare, function(eng) {
  # Special handling for glmnet
  if (eng == "glmnet") {
    spec <- linear_reg(penalty = 0) %>% set_engine(eng)  # No regularization
  } else {
    spec <- linear_reg() %>% set_engine(eng)
  }
  
  # Fit the model
  fit <- spec %>% fit(Sale_Price ~ Gr_Liv_Area + Overall_Cond, data = ames_small)
  
  # Get predictions
  preds <- predict(fit, ames_small)
  
  # Return summary
  tibble(
    engine = eng,
    rmse = sqrt(mean((ames_small$Sale_Price - preds$.pred)^2))
  )
})

bind_rows(linear_comparisons) %>%
  knitr::kable(digits = 2)

# Random forest with ranger engine
rf_ranger <- rand_forest(trees = 500) %>%
  set_engine("ranger",
             importance = "impurity",  # ranger-specific
             num.threads = 2)          # ranger-specific

# Random forest with randomForest engine
rf_randomForest <- rand_forest(trees = 500) %>%
  set_engine("randomForest",
             nodesize = 5,             # randomForest-specific
             maxnodes = 100)           # randomForest-specific

# The model specification is the same, but engine arguments differ
print(rf_ranger)
print(rf_randomForest)

# Main arguments are specified in the model function
rf_with_args <- rand_forest(
  trees = 1000,      # Number of trees
  mtry = 3,          # Variables per split
  min_n = 10         # Minimum node size
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

print(rf_with_args)

# These translate to engine-specific names
translate(rf_with_args)  # See the actual ranger call

# Start with a basic specification
base_rf <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Update with new values
updated_rf <- base_rf %>%
  set_args(trees = 2000, mtry = 5)

print(updated_rf)

# This is useful for tuning
tunable_rf <- rand_forest(
  trees = tune(),    # Mark for tuning
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

print(tunable_rf)

# Fit different models
models <- list(
  linear = linear_reg() %>% 
    set_engine("lm") %>%
    fit(Sale_Price ~ Gr_Liv_Area + Overall_Cond, data = ames_small),
  
  tree = decision_tree() %>%
    set_engine("rpart") %>%
    set_mode("regression") %>%
    fit(Sale_Price ~ Gr_Liv_Area + Overall_Cond, data = ames_small),
  
  knn = nearest_neighbor(neighbors = 5) %>%
    set_engine("kknn") %>%
    set_mode("regression") %>%
    fit(Sale_Price ~ Gr_Liv_Area + Overall_Cond, data = ames_small)
)

# All predictions have the same format
predictions <- map(models, ~ predict(., ames_small))

# Check structure - all identical!
map(predictions, str)

# For classification, we can get probabilities consistently
class_model <- logistic_reg() %>%
  set_engine("glm") %>%
  fit(expensive ~ Gr_Liv_Area + Overall_Cond, data = ames_class)

# Class predictions
class_preds <- predict(class_model, ames_class)
head(class_preds)

# Probability predictions
prob_preds <- predict(class_model, ames_class, type = "prob")
head(prob_preds)

# Ridge regression
ridge_spec <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine("glmnet")

# Lasso regression  
lasso_spec <- linear_reg(penalty = 0.1, mixture = 1) %>%
  set_engine("glmnet")

# Elastic net
elastic_spec <- linear_reg(penalty = 0.1, mixture = 0.5) %>%
  set_engine("glmnet")

# Fit and compare
regularized_models <- list(
  ridge = ridge_spec,
  lasso = lasso_spec,
  elastic = elastic_spec
) %>%
  map(~ fit(., Sale_Price ~ ., data = ames_small))

# Extract coefficients
coef_comparison <- map_df(names(regularized_models), function(model_name) {
  coefs <- regularized_models[[model_name]] %>%
    tidy() %>%
    filter(term != "(Intercept)") %>%
    mutate(model = model_name)
})

# Visualize coefficient shrinkage
ggplot(coef_comparison, aes(x = term, y = estimate, fill = model)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Coefficient Comparison: Ridge vs Lasso vs Elastic Net",
    subtitle = "Notice how Lasso sets some coefficients to exactly zero",
    x = "Variable", y = "Coefficient"
  ) +
  theme(axis.text.y = element_text(size = 8))

# XGBoost specification
xgb_spec <- boost_tree(
  trees = 1000,           # Number of trees
  tree_depth = 6,         # Maximum tree depth
  min_n = 10,             # Minimum node size
  loss_reduction = 0.01,  # Minimum loss reduction
  sample_size = 0.8,      # Subsample ratio
  learn_rate = 0.01       # Learning rate
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

print(xgb_spec)

# Fit the model
xgb_fit <- xgb_spec %>%
  fit(Sale_Price ~ ., data = ames_small)

# Feature importance
xgb_importance <- xgb_fit %>%
  extract_fit_engine() %>%
  xgboost::xgb.importance(model = .) %>%
  as_tibble()

ggplot(xgb_importance %>% head(10), 
       aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "XGBoost Feature Importance",
    subtitle = "Top 10 most important features",
    x = "Feature", y = "Importance (Gain)"
  )

# Define multiple model specifications
model_specs <- list(
  linear = linear_reg() %>% 
    set_engine("lm"),
  
  ridge = linear_reg(penalty = 0.1, mixture = 0) %>%
    set_engine("glmnet"),
  
  tree = decision_tree(tree_depth = 10) %>%
    set_engine("rpart") %>%
    set_mode("regression"),
  
  rf = rand_forest(trees = 100) %>%
    set_engine("ranger") %>%
    set_mode("regression"),
  
  knn = nearest_neighbor(neighbors = 10) %>%
    set_engine("kknn") %>%
    set_mode("regression")
)

# Fit all models
fitted_models <- map(model_specs, ~ fit(., Sale_Price ~ ., data = ames_small))

# Evaluate all models
model_evaluation <- map_df(names(fitted_models), function(model_name) {
  model <- fitted_models[[model_name]]
  
  # Get predictions
  preds <- predict(model, ames_small)$.pred
  
  # Calculate metrics
  tibble(
    model = model_name,
    rmse = sqrt(mean((ames_small$Sale_Price - preds)^2)),
    mae = mean(abs(ames_small$Sale_Price - preds)),
    r_squared = cor(ames_small$Sale_Price, preds)^2
  )
})

# Visualize comparison
model_evaluation %>%
  pivot_longer(cols = c(rmse, mae, r_squared), 
               names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = model, y = value, fill = model)) +
  geom_col() +
  facet_wrap(~metric, scales = "free_y") +
  labs(
    title = "Model Performance Comparison",
    subtitle = "Different metrics across model types",
    x = "Model", y = "Value"
  ) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

# Function to create model specs with different hyperparameters
create_rf_spec <- function(n_trees, mtry_prop, min_node) {
  rand_forest(
    trees = n_trees,
    mtry = floor(mtry_prop * ncol(ames_small) - 1),  # Convert to integer
    min_n = min_node
  ) %>%
    set_engine("ranger") %>%
    set_mode("regression")
}

# Create a grid of specifications
rf_grid <- expand_grid(
  n_trees = c(100, 500, 1000),
  mtry_prop = c(0.3, 0.5, 0.7),
  min_node = c(5, 10, 20)
)

# Create specifications
rf_specs <- pmap(rf_grid, create_rf_spec)

# Fit a subset and compare
subset_specs <- rf_specs[c(1, 14, 27)]  # Low, medium, high complexity
subset_fits <- map(subset_specs, ~ fit(., Sale_Price ~ ., data = ames_small))

# Evaluate
subset_evaluation <- map_df(1:3, function(i) {
  preds <- predict(subset_fits[[i]], ames_small)$.pred
  tibble(
    config = c("Low", "Medium", "High")[i],
    trees = rf_grid$n_trees[c(1, 14, 27)][i],
    mtry_prop = rf_grid$mtry_prop[c(1, 14, 27)][i],
    min_node = rf_grid$min_node[c(1, 14, 27)][i],
    rmse = sqrt(mean((ames_small$Sale_Price - preds)^2))
  )
})

knitr::kable(subset_evaluation, digits = 2)

# See how parsnip translates to different engines
rf_spec <- rand_forest(trees = 500, mtry = 5, min_n = 10) %>%
  set_mode("regression")

# Translation for ranger
rf_spec %>%
  set_engine("ranger") %>%
  translate()

# Translation for randomForest
rf_spec %>%
  set_engine("randomForest") %>%
  translate()

# The arguments are mapped appropriately!
# trees -> num.trees (ranger) or ntree (randomForest)
# mtry -> mtry (both)
# min_n -> min.node.size (ranger) or nodesize (randomForest)

# Start with default values
simple_rf <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Add complexity as needed
complex_rf <- rand_forest(
  trees = 1000,
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger",
             importance = "impurity",
             num.threads = parallel::detectCores() - 1) %>%
  set_mode("regression")

# Create a naming convention for your specifications
models <- list(
  # Baseline models
  baseline_mean = null_model() %>% 
    set_engine("parsnip") %>%
    set_mode("regression"),
  
  baseline_linear = linear_reg() %>%
    set_engine("lm"),
  
  # Regularized models
  reg_ridge = linear_reg(penalty = 0.1, mixture = 0) %>%
    set_engine("glmnet"),
  
  reg_lasso = linear_reg(penalty = 0.1, mixture = 1) %>%
    set_engine("glmnet"),
  
  # Tree models
  tree_single = decision_tree() %>%
    set_engine("rpart") %>%
    set_mode("regression"),
  
  tree_rf = rand_forest() %>%
    set_engine("ranger") %>%
    set_mode("regression"),
  
  tree_boost = boost_tree() %>%
    set_engine("xgboost") %>%
    set_mode("regression")
)

# Document why you chose specific values
production_rf <- rand_forest(
  trees = 500,        # Balanced accuracy vs training time
  mtry = 5,          # sqrt(p) rule for regression
  min_n = 20         # Prevent overfitting on small samples
) %>%
  set_engine("ranger",
             importance = "impurity",  # Need feature importance
             seed = 123,               # Reproducibility
             num.threads = 4) %>%      # Server has 8 cores, use half
  set_mode("regression")

# Check if required package is installed
check_model_package <- function(model_spec) {
  required_pkg <- model_spec$engine
  
  if (!requireNamespace(required_pkg, quietly = TRUE)) {
    cat("Package", required_pkg, "is not installed.\n")
    cat("Install with: install.packages('", required_pkg, "')\n", sep = "")
    return(FALSE)
  }
  
  cat("Package", required_pkg, "is available.\n")
  return(TRUE)
}

# Test
svm_spec <- svm_rbf() %>% set_engine("kernlab")
check_model_package(svm_spec)

# Not all combinations work
tryCatch({
  bad_spec <- linear_reg() %>%
    set_engine("rpart")  # Decision tree engine for linear regression?
}, error = function(e) {
  cat("Error:", e$message, "\n")
})

# Check valid combinations
show_engines("linear_reg")

# Some engines require specific arguments
tryCatch({
  bad_glmnet <- linear_reg() %>%
    set_engine("glmnet")  # Missing penalty!
  
  fit(bad_glmnet, Sale_Price ~ ., data = ames_small)
}, error = function(e) {
  cat("Error: glmnet requires penalty argument\n")
})

# Correct specification
good_glmnet <- linear_reg(penalty = 0.1) %>%
  set_engine("glmnet")

# Your solution
# Compare random forest implementations
engines <- c("ranger", "randomForest")

rf_comparison <- map_df(engines, function(eng) {
  # Skip if package not available
  if (!requireNamespace(eng, quietly = TRUE)) {
    return(tibble(engine = eng, status = "Package not installed"))
  }
  
  # Create specification
  spec <- rand_forest(trees = 100) %>%
    set_engine(eng) %>%
    set_mode("regression")
  
  # Time the fitting
  start_time <- Sys.time()
  fit <- spec %>% fit(Sale_Price ~ ., data = ames_small)
  fit_time <- as.numeric(Sys.time() - start_time, units = "secs")
  
  # Get predictions
  preds <- predict(fit, ames_small)$.pred
  
  # Return metrics
  tibble(
    engine = eng,
    fit_time = fit_time,
    rmse = sqrt(mean((ames_small$Sale_Price - preds)^2)),
    status = "Success"
  )
})

knitr::kable(rf_comparison, digits = 2)

# Your solution
# Create diverse model specifications
model_grid <- tribble(
  ~name, ~type, ~engine, ~hyperparams,
  "linear_basic", "linear_reg", "lm", list(),
  "linear_ridge", "linear_reg", "glmnet", list(penalty = 0.01, mixture = 0),
  "tree_shallow", "decision_tree", "rpart", list(tree_depth = 5),
  "tree_deep", "decision_tree", "rpart", list(tree_depth = 15),
  "rf_small", "rand_forest", "ranger", list(trees = 50),
  "rf_large", "rand_forest", "ranger", list(trees = 500),
  "knn_few", "nearest_neighbor", "kknn", list(neighbors = 3),
  "knn_many", "nearest_neighbor", "kknn", list(neighbors = 20)
)

# Function to create spec from grid row
create_spec <- function(type, engine, hyperparams) {
  # Get base specification
  spec <- switch(type,
    linear_reg = linear_reg(),
    decision_tree = decision_tree(),
    rand_forest = rand_forest(),
    nearest_neighbor = nearest_neighbor()
  )
  
  # Add hyperparameters
  if (length(hyperparams) > 0) {
    spec <- do.call(set_args, c(list(spec), hyperparams))
  }
  
  # Set engine and mode
  spec %>%
    set_engine(engine) %>%
    set_mode("regression")
}

# Create all specifications
all_specs <- pmap(model_grid %>% select(-name), create_spec)
names(all_specs) <- model_grid$name

# Fit and evaluate a few
sample_models <- all_specs[c("linear_basic", "tree_deep", "rf_large")]
sample_fits <- map(sample_models, ~ fit(., Sale_Price ~ ., data = ames_small))

# Quick evaluation
map_dbl(sample_fits, function(fit) {
  preds <- predict(fit, ames_small)$.pred
  sqrt(mean((ames_small$Sale_Price - preds)^2))
})

# Your solution
# Random forest with different engine features

# Ranger with importance
rf_ranger_imp <- rand_forest(trees = 100) %>%
  set_engine("ranger", 
             importance = "permutation",
             keep.inbag = TRUE) %>%  # For prediction intervals
  set_mode("regression") %>%
  fit(Sale_Price ~ ., data = ames_small)

# Extract ranger-specific features
ranger_importance <- rf_ranger_imp %>%
  extract_fit_engine() %>%
  .$variable.importance %>%
  sort(decreasing = TRUE) %>%
  head(10)

print("Ranger Variable Importance:")
print(ranger_importance)

# RandomForest with proximity
if (requireNamespace("randomForest", quietly = TRUE)) {
  rf_rf_prox <- rand_forest(trees = 100) %>%
    set_engine("randomForest",
               proximity = TRUE,  # Calculate proximity matrix
               keep.forest = TRUE) %>%
    set_mode("regression") %>%
    fit(Sale_Price ~ ., data = ames_small)
  
  # The proximity matrix shows similarity between observations
  # This is useful for clustering and outlier detection
  cat("\nrandomForest can provide proximity matrix for clustering\n")
}
