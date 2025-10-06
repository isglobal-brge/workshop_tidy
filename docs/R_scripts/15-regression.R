library(tidymodels)
library(tidyverse)
library(modeldata)
library(vip)
library(glmnet)
library(corrplot)
library(patchwork)
library(broom)
library(performance)

# Set theme and seed for reproducibility
theme_set(theme_minimal())
set.seed(123)

# Load datasets
data(ames)        # House prices dataset
data(concrete)    # Concrete strength dataset
data(Chicago)     # Chicago ridership data

# Quick overview of our main dataset
glimpse(ames)

# Create simple example data
set.seed(123)
simple_data <- tibble(
  x = seq(1, 10, length.out = 50),
  y_true = 2 + 3 * x,  # True relationship
  y = y_true + rnorm(50, sd = 3)  # Add noise
)

# Fit simple linear regression
simple_lm <- lm(y ~ x, data = simple_data)

# Add predictions and residuals
simple_data <- simple_data %>%
  mutate(
    y_pred = predict(simple_lm),
    residual = y - y_pred
  )

# Visualize OLS concept
p1 <- ggplot(simple_data, aes(x = x)) +
  geom_segment(aes(xend = x, y = y, yend = y_pred), 
               color = "red", alpha = 0.5) +
  geom_point(aes(y = y), size = 2) +
  geom_line(aes(y = y_pred), color = "blue", linewidth = 1.2) +
  geom_line(aes(y = y_true), color = "green", linetype = "dashed") +
  labs(
    title = "Ordinary Least Squares Visualization",
    subtitle = "Red lines show residuals (errors) being minimized",
    x = "Predictor (X)",
    y = "Response (Y)"
  ) +
  annotate("text", x = 8, y = 15, label = "True relationship", 
           color = "green", size = 4) +
  annotate("text", x = 8, y = 25, label = "Fitted line", 
           color = "blue", size = 4)

# Residual distribution
p2 <- ggplot(simple_data, aes(x = residual)) +
  geom_histogram(bins = 20, fill = "lightblue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Distribution of Residuals",
    subtitle = "Should be centered at zero and approximately normal",
    x = "Residual",
    y = "Count"
  )

p1 + p2

# Prepare Ames data for simple example
ames_simple <- ames %>%
  select(Sale_Price, Gr_Liv_Area, Year_Built, Overall_Cond) %>%
  drop_na()

# Fit model
ames_lm <- lm(Sale_Price ~ Gr_Liv_Area + Year_Built + Overall_Cond, 
              data = ames_simple)

# Create diagnostic plots
par(mfrow = c(2, 2))
plot(ames_lm)

# Clean and prepare the Ames data
ames_clean <- ames %>%
  mutate(Sale_Price = log10(Sale_Price)) %>%  # Log transform for better distribution
  select(-contains("_Condition"))  # Remove some problematic variables

# Split the data
ames_split <- initial_split(ames_clean, prop = 0.75, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# Create preprocessing recipe
ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  # Remove variables with near-zero variance
  step_nzv(all_predictors()) %>%
  # Impute missing values
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # Create dummy variables for categorical predictors
  step_dummy(all_nominal_predictors()) %>%
  # Normalize numeric predictors
  step_normalize(all_numeric_predictors())

# Check the recipe
prep(ames_recipe) %>%
  bake(new_data = NULL) %>%
  glimpse()

# Specify linear regression model
lm_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

# Create workflow
lm_workflow <- workflow() %>%
  add_recipe(ames_recipe) %>%
  add_model(lm_spec)

# Fit the model
lm_fit <- lm_workflow %>%
  fit(ames_train)

# Extract and examine coefficients
lm_coefs <- lm_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate))) %>%
  head(20)

# Visualize top coefficients
ggplot(lm_coefs, aes(x = reorder(term, estimate), y = estimate)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 20 Most Influential Features in Linear Regression",
    subtitle = "Coefficients represent change in log10(Sale_Price) per unit change in feature",
    x = "Feature",
    y = "Coefficient"
  )

# Generate non-linear data
set.seed(123)
nonlinear_data <- tibble(
  x = seq(0, 10, length.out = 100),
  y_true = 10 + 5*x - 0.5*x^2 + 0.02*x^3,
  y = y_true + rnorm(100, sd = 5)
)

# Fit models with different polynomial degrees
degrees <- c(1, 2, 3, 5, 10)
poly_models <- map(degrees, function(d) {
  lm(y ~ poly(x, degree = d, raw = TRUE), data = nonlinear_data)
})

# Generate predictions
poly_predictions <- map2_df(poly_models, degrees, function(model, d) {
  nonlinear_data %>%
    mutate(
      y_pred = predict(model),
      degree = paste("Degree", d)
    )
})

# Visualize
ggplot(poly_predictions, aes(x = x)) +
  geom_point(aes(y = y), alpha = 0.3, data = nonlinear_data) +
  geom_line(aes(y = y_pred, color = degree), linewidth = 1.2) +
  geom_line(aes(y = y_true), color = "black", linetype = "dashed", 
            data = nonlinear_data, linewidth = 1) +
  facet_wrap(~degree, ncol = 3) +
  scale_color_viridis_d() +
  labs(
    title = "Polynomial Regression with Different Degrees",
    subtitle = "Black dashed line shows true relationship. Higher degrees can lead to overfitting.",
    x = "X",
    y = "Y"
  ) +
  theme(legend.position = "none")

# Create recipe with polynomial features
poly_recipe <- recipe(Sale_Price ~ Gr_Liv_Area + Lot_Area, 
                      data = ames_train) %>%
  step_poly(Gr_Liv_Area, degree = 2) %>%
  step_poly(Lot_Area, degree = 2) %>%
  step_interact(terms = ~ Gr_Liv_Area_poly_1:Lot_Area_poly_1)

# Fit model with polynomial features
poly_workflow <- workflow() %>%
  add_recipe(poly_recipe) %>%
  add_model(lm_spec)

poly_fit <- poly_workflow %>%
  fit(ames_train)

# Compare with simple linear model
simple_recipe <- recipe(Sale_Price ~ Gr_Liv_Area + Lot_Area, 
                       data = ames_train)

simple_workflow <- workflow() %>%
  add_recipe(simple_recipe) %>%
  add_model(lm_spec)

simple_fit <- simple_workflow %>%
  fit(ames_train)

# Evaluate both models
models_comparison <- tibble(
  model = c("Linear", "Polynomial"),
  workflow = list(simple_workflow, poly_workflow)
) %>%
  mutate(
    fit = map(workflow, ~ fit(., ames_train)),
    train_pred = map(fit, ~ predict(., ames_train)),
    test_pred = map(fit, ~ predict(., ames_test))
  )

# Calculate metrics
comparison_metrics <- models_comparison %>%
  mutate(
    train_metrics = map2(train_pred, list(ames_train), ~ {
      bind_cols(.x, .y) %>%
        metrics(truth = Sale_Price, estimate = .pred)
    }),
    test_metrics = map2(test_pred, list(ames_test), ~ {
      bind_cols(.x, .y) %>%
        metrics(truth = Sale_Price, estimate = .pred)
    })
  ) %>%
  select(model, train_metrics, test_metrics) %>%
  pivot_longer(c(train_metrics, test_metrics), 
               names_to = "dataset", values_to = "metrics") %>%
  unnest(metrics)

# Visualize comparison
ggplot(comparison_metrics, 
       aes(x = model, y = .estimate, fill = dataset)) +
  geom_col(position = "dodge") +
  facet_wrap(~.metric, scales = "free_y") +
  scale_fill_manual(values = c("train_metrics" = "lightblue", 
                               "test_metrics" = "darkblue")) +
  labs(
    title = "Linear vs Polynomial Model Comparison",
    subtitle = "Watch for overfitting: train performance much better than test",
    y = "Metric Value"
  )

# Prepare data for regularization
ames_reg <- ames_train %>%
  select(Sale_Price, Gr_Liv_Area, Garage_Area, Year_Built, 
         Lot_Area, Total_Bsmt_SF, First_Flr_SF) %>%
  drop_na()

# Create model matrix
X <- model.matrix(Sale_Price ~ . - 1, data = ames_reg)
y <- ames_reg$Sale_Price

# Fit Ridge regression
ridge_fit <- glmnet(X, y, alpha = 0)  # alpha = 0 for Ridge

# Fit Lasso regression
lasso_fit <- glmnet(X, y, alpha = 1)  # alpha = 1 for Lasso

# Fit Elastic Net
elastic_fit <- glmnet(X, y, alpha = 0.5)  # alpha = 0.5 for 50/50 mix

# Create coefficient path plots
par(mfrow = c(1, 3))
plot(ridge_fit, xvar = "lambda", main = "Ridge Regression Path")
plot(lasso_fit, xvar = "lambda", main = "Lasso Regression Path")
plot(elastic_fit, xvar = "lambda", main = "Elastic Net Path")

# Create a more complete recipe
reg_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_nzv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Ridge regression
ridge_spec <- linear_reg(penalty = 0.01, mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Lasso regression
lasso_spec <- linear_reg(penalty = 0.01, mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Elastic Net
elastic_spec <- linear_reg(penalty = 0.01, mixture = 0.5) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Create workflows
ridge_wf <- workflow() %>%
  add_recipe(reg_recipe) %>%
  add_model(ridge_spec)

lasso_wf <- workflow() %>%
  add_recipe(reg_recipe) %>%
  add_model(lasso_spec)

elastic_wf <- workflow() %>%
  add_recipe(reg_recipe) %>%
  add_model(elastic_spec)

# Fit all models
ridge_fit <- ridge_wf %>% fit(ames_train)
lasso_fit <- lasso_wf %>% fit(ames_train)
elastic_fit <- elastic_wf %>% fit(ames_train)

# Compare number of non-zero coefficients
n_features <- tibble(
  Model = c("Ridge", "Lasso", "Elastic Net"),
  `Non-zero Coefficients` = c(
    sum(tidy(ridge_fit)$estimate != 0),
    sum(tidy(lasso_fit)$estimate != 0),
    sum(tidy(elastic_fit)$estimate != 0)
  ),
  `Total Features` = c(
    nrow(tidy(ridge_fit)),
    nrow(tidy(lasso_fit)),
    nrow(tidy(elastic_fit))
  )
)

knitr::kable(n_features)

# Fit a simple regression tree
tree_spec <- decision_tree(
  cost_complexity = 0.01,
  tree_depth = 4,
  min_n = 20
) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_fit <- tree_spec %>%
  fit(Sale_Price ~ Gr_Liv_Area + Year_Built, data = ames_train)

# Visualize the tree
if (require(rpart.plot, quietly = TRUE)) {
  rpart.plot(tree_fit$fit, type = 4, extra = 1, roundint = FALSE)
}

# Create prediction surface
grid <- expand_grid(
  Gr_Liv_Area = seq(min(ames_train$Gr_Liv_Area, na.rm = TRUE),
                     max(ames_train$Gr_Liv_Area, na.rm = TRUE),
                     length.out = 100),
  Year_Built = seq(min(ames_train$Year_Built, na.rm = TRUE),
                   max(ames_train$Year_Built, na.rm = TRUE),
                   length.out = 100)
)

grid_pred <- tree_fit %>%
  predict(grid) %>%
  bind_cols(grid)

# Visualize prediction surface
ggplot(grid_pred, aes(x = Gr_Liv_Area, y = Year_Built, fill = .pred)) +
  geom_tile() +
  scale_fill_viridis_c() +
  geom_point(data = sample_n(ames_train, 200), 
             aes(fill = NULL, color = Sale_Price), 
             size = 1, alpha = 0.5) +
  scale_color_viridis_c() +
  labs(
    title = "Regression Tree Prediction Surface",
    subtitle = "Notice the rectangular regions - characteristic of tree-based methods",
    x = "Living Area (sq ft)",
    y = "Year Built",
    fill = "Predicted\nlog(Price)",
    color = "Actual\nlog(Price)"
  )

# Random Forest specification
rf_spec <- rand_forest(
  trees = 500,
  mtry = 10,
  min_n = 5
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

# Create workflow
rf_workflow <- workflow() %>%
  add_recipe(reg_recipe) %>%
  add_model(rf_spec)

# Fit with cross-validation for better evaluation
set.seed(123)
ames_folds <- vfold_cv(ames_train, v = 5)

rf_cv <- rf_workflow %>%
  fit_resamples(
    resamples = ames_folds,
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae),
    control = control_resamples(save_pred = TRUE)
  )

# Show performance
collect_metrics(rf_cv) %>%
  knitr::kable(digits = 4)

# Fit final model for feature importance
rf_final <- rf_workflow %>%
  fit(ames_train)

# Extract and plot feature importance
rf_importance <- rf_final %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20)

rf_importance +
  labs(title = "Top 20 Most Important Features in Random Forest",
       subtitle = "Based on impurity reduction")

# Get predictions from our linear model
lm_pred <- lm_fit %>%
  predict(ames_test) %>%
  bind_cols(ames_test) %>%
  mutate(
    residual = Sale_Price - .pred,
    std_residual = residual / sd(residual)
  )

# Create diagnostic plots
p1 <- ggplot(lm_pred, aes(x = .pred, y = residual)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(
    title = "Residuals vs Fitted Values",
    subtitle = "Look for patterns - should be random scatter around zero",
    x = "Fitted Values",
    y = "Residuals"
  )

p2 <- ggplot(lm_pred, aes(sample = std_residual)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(
    title = "Q-Q Plot",
    subtitle = "Check normality - points should follow the line",
    x = "Theoretical Quantiles",
    y = "Standardized Residuals"
  )

p3 <- ggplot(lm_pred, aes(x = .pred, y = sqrt(abs(std_residual)))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(
    title = "Scale-Location Plot",
    subtitle = "Check homoscedasticity - spread should be constant",
    x = "Fitted Values",
    y = "√|Standardized Residuals|"
  )

p4 <- ggplot(lm_pred, aes(x = residual)) +
  geom_histogram(bins = 30, fill = "lightblue", color = "black", alpha = 0.7) +
  geom_density(aes(y = after_stat(count)), color = "red", linewidth = 1) +
  labs(
    title = "Distribution of Residuals",
    subtitle = "Should be approximately normal",
    x = "Residual",
    y = "Count"
  )

(p1 + p2) / (p3 + p4)

# Compare multiple models using cross-validation
models_list <- list(
  "Linear" = lm_spec,
  "Ridge" = ridge_spec,
  "Lasso" = lasso_spec,
  "Random Forest" = rf_spec,
  "Decision Tree" = tree_spec
)

# Evaluate all models
cv_results <- map_df(names(models_list), function(model_name) {
  wf <- workflow() %>%
    add_recipe(reg_recipe) %>%
    add_model(models_list[[model_name]])
  
  fit_resamples(
    wf,
    resamples = ames_folds,
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
  ) %>%
    collect_metrics() %>%
    mutate(model = model_name)
})

# Visualize comparison
ggplot(cv_results, aes(x = model, y = mean, fill = model)) +
  geom_col() +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err), 
                width = 0.2) +
  facet_wrap(~.metric, scales = "free_y") +
  scale_fill_viridis_d() +
  labs(
    title = "Model Comparison via 5-Fold Cross-Validation",
    subtitle = "Error bars show standard error across folds",
    y = "Metric Value"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Check correlation among numeric predictors
numeric_features <- ames_train %>%
  select(where(is.numeric)) %>%
  select(-Sale_Price) %>%
  drop_na()

cor_matrix <- cor(numeric_features)

# Find highly correlated pairs
high_cor <- which(abs(cor_matrix) > 0.8 & cor_matrix != 1, arr.ind = TRUE)
high_cor_pairs <- tibble(
  var1 = rownames(cor_matrix)[high_cor[,1]],
  var2 = colnames(cor_matrix)[high_cor[,2]],
  correlation = cor_matrix[high_cor]
) %>%
  filter(var1 < var2) %>%  # Remove duplicates
  arrange(desc(abs(correlation)))

print("Highly correlated variable pairs:")
head(high_cor_pairs, 10) %>% knitr::kable(digits = 3)

# Visualize correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.6, tl.col = "black",
         diag = FALSE, addCoef.col = "black", number.cex = 0.4)

# Example with original scale prices (not log-transformed)
ames_original <- ames %>%
  select(Sale_Price, Gr_Liv_Area, Year_Built, Overall_Cond) %>%
  drop_na()

# Fit model on original scale
original_fit <- lm(Sale_Price ~ ., data = ames_original)

# Fit model on log scale
log_fit <- lm(log(Sale_Price) ~ ., data = ames_original)

# Compare residual plots
orig_pred <- tibble(
  fitted = fitted(original_fit),
  residual = residuals(original_fit),
  model = "Original Scale"
)

log_pred <- tibble(
  fitted = fitted(log_fit),
  residual = residuals(log_fit),
  model = "Log Scale"
)

combined_pred <- bind_rows(orig_pred, log_pred)

ggplot(combined_pred, aes(x = fitted, y = residual)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  facet_wrap(~model, scales = "free") +
  labs(
    title = "Effect of Log Transformation on Heteroscedasticity",
    subtitle = "Log transformation often stabilizes variance",
    x = "Fitted Values",
    y = "Residuals"
  )

# Create a complete modeling pipeline
# 1. Data preparation
concrete_clean <- concrete %>%
  drop_na()

concrete_split <- initial_split(concrete_clean, prop = 0.8)
concrete_train <- training(concrete_split)
concrete_test <- testing(concrete_split)

# 2. Exploratory data analysis
concrete_summary <- concrete_train %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  summarise(
    mean = mean(value),
    sd = sd(value),
    min = min(value),
    max = max(value)
  )

print("Dataset summary:")
concrete_summary %>% knitr::kable(digits = 2)

# 3. Create recipe with feature engineering
concrete_recipe <- recipe(compressive_strength ~ ., data = concrete_train) %>%
  step_normalize(all_predictors()) %>%
  step_poly(age, degree = 2) %>%
  step_interact(terms = ~ cement:water)

# 4. Specify multiple models
models <- list(
  linear = linear_reg() %>% set_engine("lm"),
  ridge = linear_reg(penalty = tune(), mixture = 0) %>% set_engine("glmnet"),
  rf = rand_forest(mtry = tune(), trees = 500, min_n = tune()) %>% 
    set_engine("ranger") %>% set_mode("regression")
)

# 5. Create workflows
workflows <- map(models, function(model) {
  workflow() %>%
    add_recipe(concrete_recipe) %>%
    add_model(model)
})

# 6. Tune hyperparameters for models that need it
concrete_folds <- vfold_cv(concrete_train, v = 5)

# Tune Ridge
ridge_grid <- grid_regular(penalty(), levels = 10)
ridge_tune <- workflows$ridge %>%
  tune_grid(
    resamples = concrete_folds,
    grid = ridge_grid,
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq)
  )

# Tune Random Forest
rf_grid <- grid_regular(
  mtry(range = c(2, 7)),
  min_n(),
  levels = 5
)
rf_tune <- workflows$rf %>%
  tune_grid(
    resamples = concrete_folds,
    grid = rf_grid,
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq)
  )

# 7. Select best models
best_ridge <- select_best(ridge_tune, metric = "rmse")
best_rf <- select_best(rf_tune, metric = "rmse")

# 8. Finalize workflows
final_linear <- workflows$linear
final_ridge <- workflows$ridge %>% finalize_workflow(best_ridge)
final_rf <- workflows$rf %>% finalize_workflow(best_rf)

# 9. Fit final models and evaluate
final_fits <- list(
  linear = final_linear %>% fit(concrete_train),
  ridge = final_ridge %>% fit(concrete_train),
  rf = final_rf %>% fit(concrete_train)
)

# 10. Make predictions and evaluate
test_results <- map_df(names(final_fits), function(model_name) {
  final_fits[[model_name]] %>%
    predict(concrete_test) %>%
    bind_cols(concrete_test) %>%
    metrics(truth = compressive_strength, estimate = .pred) %>%
    mutate(model = model_name)
})

# Visualize final comparison
ggplot(test_results, aes(x = model, y = .estimate, fill = model)) +
  geom_col() +
  facet_wrap(~.metric, scales = "free_y") +
  scale_fill_viridis_d() +
  labs(
    title = "Final Model Comparison on Test Set",
    subtitle = "Concrete compressive strength prediction",
    y = "Metric Value"
  ) +
  theme(legend.position = "none")

# Your solution
# Use the Ames dataset to predict Sale_Price
# Create an elastic net model with tuned penalty and mixture

# Simpler recipe for exercise
elastic_recipe <- recipe(Sale_Price ~ Gr_Liv_Area + Year_Built + Overall_Cond + 
                        Neighborhood + Total_Bsmt_SF, data = ames_train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Elastic net specification with tuning
elastic_tune_spec <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Workflow
elastic_tune_wf <- workflow() %>%
  add_recipe(elastic_recipe) %>%
  add_model(elastic_tune_spec)

# Tuning grid
elastic_grid <- grid_regular(
  penalty(range = c(-3, 0)),
  mixture(range = c(0, 1)),
  levels = c(10, 5)
)

# Create resamples
ames_folds <- vfold_cv(ames_train, v = 5)

# Tune the model
elastic_tuned <- elastic_tune_wf %>%
  tune_grid(
    resamples = ames_folds,
    grid = elastic_grid,
    metrics = yardstick::metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)
  )

# Visualize tuning results
autoplot(elastic_tuned) +
  labs(title = "Elastic Net Tuning Results",
       subtitle = "Performance across different penalty and mixture values")

# Select and fit best model
best_elastic <- select_best(elastic_tuned, metric = "rmse")
final_elastic <- elastic_tune_wf %>%
  finalize_workflow(best_elastic) %>%
  fit(ames_train)

# Evaluate on test set
elastic_test_pred <- final_elastic %>%
  predict(ames_test) %>%
  bind_cols(ames_test)

elastic_metrics <- elastic_test_pred %>%
  metrics(truth = Sale_Price, estimate = .pred)

print("Best hyperparameters:")
print(best_elastic)
print("Test set performance:")
print(elastic_metrics)

# Your solution
# Create a problematic model and fix it

# Intentionally problematic model (using highly correlated predictors)
problem_data <- ames_train %>%
  select(Sale_Price, Gr_Liv_Area, Total_Bsmt_SF, First_Flr_SF, 
         Second_Flr_SF, Garage_Area, Garage_Cars) %>%
  drop_na()

# Check correlations
cor(problem_data[,-1]) %>%
  corrplot(method = "number", type = "upper")

# Fit problematic model
problem_fit <- lm(Sale_Price ~ ., data = problem_data)
summary(problem_fit)

# Note the high VIF (Variance Inflation Factor)
if (require(car, quietly = TRUE)) {
  vif_values <- car::vif(problem_fit)
  print("Variance Inflation Factors:")
  print(vif_values)
}

# Fix 1: Remove highly correlated variables
fixed_data1 <- problem_data %>%
  select(-Garage_Cars, -Second_Flr_SF)  # Remove redundant variables

fixed_fit1 <- lm(Sale_Price ~ ., data = fixed_data1)

# Fix 2: Use PCA
pca_recipe <- recipe(Sale_Price ~ ., data = problem_data) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = 4)

pca_fit <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(lm_spec) %>%
  fit(problem_data)

# Fix 3: Use Ridge regression
ridge_fix <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_engine("glmnet") %>%
  fit(Sale_Price ~ ., data = problem_data)

# Compare solutions
cat("Original model R-squared:", summary(problem_fit)$r.squared, "\n")
cat("Fixed model R-squared:", summary(fixed_fit1)$r.squared, "\n")

# Your solution
# Create synthetic data with non-linear relationship
set.seed(456)
nonlinear_ex <- tibble(
  x = seq(0, 10, length.out = 200),
  y = 5 + 3*sin(x) + 0.5*x^2 - 0.05*x^3 + rnorm(200, sd = 1)
)

# Visualize the relationship
ggplot(nonlinear_ex, aes(x = x, y = y)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(title = "Non-linear Relationship to Model")

# Compare different approaches
# 1. Linear model (will underfit)
linear_ex <- lm(y ~ x, data = nonlinear_ex)

# 2. Polynomial regression
poly_ex <- lm(y ~ poly(x, 5), data = nonlinear_ex)

# 3. Spline regression
spline_ex <- lm(y ~ splines::ns(x, df = 5), data = nonlinear_ex)

# 4. Random Forest
rf_ex <- rand_forest(trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("regression") %>%
  fit(y ~ x, data = nonlinear_ex)

# Generate predictions
predictions_ex <- nonlinear_ex %>%
  mutate(
    linear = predict(linear_ex, nonlinear_ex),
    polynomial = predict(poly_ex, nonlinear_ex),
    spline = predict(spline_ex, nonlinear_ex),
    rf = predict(rf_ex, nonlinear_ex)$.pred
  ) %>%
  pivot_longer(c(linear, polynomial, spline, rf), 
               names_to = "model", values_to = "prediction")

# Visualize all models
ggplot(predictions_ex, aes(x = x)) +
  geom_point(aes(y = y), alpha = 0.2) +
  geom_line(aes(y = prediction, color = model), linewidth = 1.2) +
  facet_wrap(~model) +
  scale_color_viridis_d() +
  labs(
    title = "Comparison of Methods for Non-linear Relationships",
    subtitle = "Different approaches to capturing non-linearity"
  ) +
  theme(legend.position = "none")

# Calculate performance
performance_ex <- predictions_ex %>%
  group_by(model) %>%
  summarise(
    rmse = sqrt(mean((y - prediction)^2)),
    mae = mean(abs(y - prediction)),
    r_squared = cor(y, prediction)^2
  )

performance_ex %>% knitr::kable(digits = 3)
