library(tidymodels)
library(tidyverse)
library(lubridate)
library(textrecipes)
library(themis)
library(corrplot)

# Set theme and seed
theme_set(theme_minimal())
set.seed(123)

# Load example datasets
data(ames)
data(credit_data)

# Create a simple example to show feature engineering impact
simple_data <- tibble(
  sale_date = seq(as.Date("2020-01-01"), as.Date("2022-12-31"), by = "day"),
  temperature = 50 + 30 * sin(2 * pi * as.numeric(sale_date) / 365) + rnorm(length(sale_date), 0, 5),
  sales = 1000 + 200 * sin(2 * pi * as.numeric(sale_date) / 365) + 
          100 * (wday(sale_date) %in% c(1, 7)) +  # Weekend boost
          rnorm(length(sale_date), 0, 50)
)

# Without feature engineering - just using date as numeric
bad_model <- lm(sales ~ as.numeric(sale_date), data = simple_data)

# With feature engineering
good_data <- simple_data %>%
  mutate(
    month = month(sale_date),
    day_of_week = wday(sale_date, label = TRUE),
    is_weekend = wday(sale_date) %in% c(1, 7),
    quarter = quarter(sale_date),
    days_since_start = as.numeric(sale_date - min(sale_date))
  )

good_model <- lm(sales ~ month + is_weekend + temperature + days_since_start, 
                 data = good_data)

# Compare R-squared
tibble(
  Model = c("Without Feature Engineering", "With Feature Engineering"),
  `R-squared` = c(summary(bad_model)$r.squared, summary(good_model)$r.squared)
) %>%
  knitr::kable(digits = 3)

# Prepare the Ames housing data
ames_train <- ames %>%
  filter(Sale_Price > 0) %>%
  sample_frac(0.8)

ames_test <- ames %>%
  filter(Sale_Price > 0) %>%
  anti_join(ames_train)

# Create a basic recipe
basic_recipe <- recipe(Sale_Price ~ Lot_Area + Year_Built + Overall_Cond, 
                       data = ames_train)

# View the recipe
basic_recipe

# Enhanced recipe with multiple steps
enhanced_recipe <- recipe(Sale_Price ~ Lot_Area + Year_Built + Overall_Cond + 
                          Neighborhood + Gr_Liv_Area, 
                          data = ames_train) %>%
  # Step 1: Log transform the outcome
  step_log(Sale_Price) %>%
  # Step 2: Create a new feature
  step_mutate(House_Age = 2010 - Year_Built) %>%
  # Step 3: Remove the original Year_Built
  step_rm(Year_Built) %>%
  # Step 4: Normalize numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  # Step 5: Create dummy variables for categorical predictors
  step_dummy(all_nominal_predictors())

enhanced_recipe

# Prepare the recipe (learn parameters from training data)
prepped_recipe <- prep(enhanced_recipe, training = ames_train)

# See what was learned
prepped_recipe

# Apply to training data
baked_train <- bake(prepped_recipe, new_data = NULL)  # NULL means use training data
glimpse(baked_train)

# Apply to test data
baked_test <- bake(prepped_recipe, new_data = ames_test)

# Check that dimensions match (except for rows)
tibble(
  Dataset = c("Training", "Test"),
  Rows = c(nrow(baked_train), nrow(baked_test)),
  Columns = c(ncol(baked_train), ncol(baked_test))
) %>%
  knitr::kable()

# Create example data with different scales
scaling_demo <- tibble(
  feature_A = rnorm(1000, mean = 100, sd = 15),      # Normal, mean=100
  feature_B = rexp(1000, rate = 0.01),               # Exponential, right-skewed
  feature_C = runif(1000, min = 0, max = 1),         # Uniform, 0-1 range
  feature_D = rlnorm(1000, meanlog = 10, sdlog = 2)  # Log-normal, very large
)

# Different scaling recipes
scaling_recipes <- list(
  original = recipe(~ ., data = scaling_demo),
  
  normalized = recipe(~ ., data = scaling_demo) %>%
    step_normalize(all_predictors()),
  
  range = recipe(~ ., data = scaling_demo) %>%
    step_range(all_predictors(), min = 0, max = 1),
  
  robust = recipe(~ ., data = scaling_demo) %>%
    step_center(all_predictors()) %>%  # Center using median
    step_scale(all_predictors())       # Scale using standard deviation
)

# Apply each recipe
scaled_data <- map_df(names(scaling_recipes), function(name) {
  scaling_recipes[[name]] %>%
    prep() %>%
    bake(new_data = NULL) %>%
    mutate(scaling_method = name) %>%
    pivot_longer(cols = -scaling_method, 
                 names_to = "feature", 
                 values_to = "value")
})

# Visualize the effects
ggplot(scaled_data, aes(x = value, fill = scaling_method)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_grid(scaling_method ~ feature, scales = "free") +
  scale_fill_viridis_d() +
  labs(
    title = "Effects of Different Scaling Methods",
    subtitle = "Each method has different properties and use cases",
    x = "Scaled Value",
    y = "Count"
  ) +
  theme(legend.position = "none")

# Create skewed data
skewed_data <- tibble(
  mild_skew = rgamma(1000, shape = 2, rate = 0.5),
  moderate_skew = rlnorm(1000, meanlog = 0, sdlog = 1),
  severe_skew = rexp(1000, rate = 0.1),
  outcome = rnorm(1000)
)

# Different transformation recipes
transform_recipes <- list(
  original = recipe(outcome ~ ., data = skewed_data),
  
  log = recipe(outcome ~ ., data = skewed_data) %>%
    step_log(all_predictors(), offset = 1),  # offset prevents log(0)
  
  sqrt = recipe(outcome ~ ., data = skewed_data) %>%
    step_sqrt(all_predictors()),
  
  yeo_johnson = recipe(outcome ~ ., data = skewed_data) %>%
    step_YeoJohnson(all_predictors()),  # Automatic optimal transformation
  
  box_cox = recipe(outcome ~ ., data = skewed_data) %>%
    step_BoxCox(all_predictors())  # Requires positive values
)

# Apply transformations and calculate skewness
skewness_comparison <- map_df(names(transform_recipes), function(name) {
  transformed <- transform_recipes[[name]] %>%
    prep() %>%
    bake(new_data = NULL)
  
  tibble(
    method = name,
    mild_skew = moments::skewness(transformed$mild_skew),
    moderate_skew = moments::skewness(transformed$moderate_skew),
    severe_skew = moments::skewness(transformed$severe_skew)
  )
})

# Display results
skewness_comparison %>%
  pivot_longer(cols = -method, names_to = "feature", values_to = "skewness") %>%
  ggplot(aes(x = method, y = abs(skewness), fill = method)) +
  geom_col() +
  facet_wrap(~feature) +
  scale_fill_viridis_d() +
  labs(
    title = "Effectiveness of Different Transformations on Skewed Data",
    subtitle = "Lower absolute skewness is better (closer to normal distribution)",
    x = "Transformation Method",
    y = "Absolute Skewness"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Example with different types of categorical variables
cat_data <- tibble(
  color = factor(c("red", "blue", "green", "red", "blue")),
  size = factor(c("S", "M", "L", "XL", "M"), 
                levels = c("S", "M", "L", "XL"), ordered = TRUE),
  quality = factor(c("good", "bad", "excellent", "good", "bad")),
  outcome = c(10, 15, 20, 12, 14)
)

# Basic dummy encoding
dummy_recipe <- recipe(outcome ~ ., data = cat_data) %>%
  step_dummy(all_nominal_predictors())

dummy_result <- dummy_recipe %>%
  prep() %>%
  bake(new_data = NULL)

dummy_result

# Create high-cardinality example
high_card_data <- ames_train %>%
  select(Sale_Price, Neighborhood, MS_SubClass) %>%
  mutate(
    Neighborhood_Freq = n(),
    .by = Neighborhood
  )

# Different encoding strategies
encoding_recipes <- list(
  # Standard dummy encoding
  dummy = recipe(Sale_Price ~ Neighborhood, data = high_card_data) %>%
    step_dummy(Neighborhood),
  
  # Frequency encoding
  frequency = recipe(Sale_Price ~ Neighborhood, data = high_card_data) %>%
    step_mutate(Neighborhood_Freq = n(), .by = Neighborhood) %>%
    step_rm(Neighborhood),
  
  # Target encoding (mean of target for each category)
  target = recipe(Sale_Price ~ Neighborhood, data = high_card_data) %>%
    step_mutate(
      Neighborhood_Mean = mean(Sale_Price, na.rm = TRUE),
      .by = Neighborhood
    ) %>%
    step_rm(Neighborhood),
  
  # Lumping rare categories
  lumped = recipe(Sale_Price ~ Neighborhood, data = high_card_data) %>%
    step_other(Neighborhood, threshold = 0.05) %>%  # Combine rare levels
    step_dummy(Neighborhood)
)

# Compare number of features created
feature_counts <- map_df(names(encoding_recipes), function(name) {
  n_features <- encoding_recipes[[name]] %>%
    prep() %>%
    bake(new_data = NULL) %>%
    select(-Sale_Price) %>%
    ncol()
  
  tibble(
    method = name,
    n_features = n_features
  )
})

feature_counts %>%
  ggplot(aes(x = method, y = n_features, fill = method)) +
  geom_col() +
  geom_text(aes(label = n_features), vjust = -0.5) +
  scale_fill_viridis_d() +
  labs(
    title = "Feature Count with Different Encoding Methods",
    subtitle = "High-cardinality categorical variables can create many features",
    x = "Encoding Method",
    y = "Number of Features"
  ) +
  theme(legend.position = "none")

# Generate data with interaction effect
set.seed(123)
interaction_data <- tibble(
  x1 = runif(500, 0, 10),
  x2 = runif(500, 0, 10),
  # True relationship includes interaction
  y = 10 + 2*x1 + 3*x2 + 0.5*x1*x2 + rnorm(500, 0, 2)
)

# Models with and without interaction
no_interaction_recipe <- recipe(y ~ x1 + x2, data = interaction_data)

with_interaction_recipe <- recipe(y ~ x1 + x2, data = interaction_data) %>%
  step_interact(terms = ~ x1:x2)

# Fit both models
no_int_fit <- workflow() %>%
  add_recipe(no_interaction_recipe) %>%
  add_model(linear_reg()) %>%
  fit(interaction_data)

with_int_fit <- workflow() %>%
  add_recipe(with_interaction_recipe) %>%
  add_model(linear_reg()) %>%
  fit(interaction_data)

# Create prediction surface
grid <- expand_grid(
  x1 = seq(0, 10, length.out = 50),
  x2 = seq(0, 10, length.out = 50)
)

grid_no_int <- grid %>%
  mutate(
    prediction = predict(no_int_fit, grid)$.pred,
    model = "Without Interaction"
  )

grid_with_int <- grid %>%
  mutate(
    prediction = predict(with_int_fit, grid)$.pred,
    model = "With Interaction"
  )

# Visualize the difference
bind_rows(grid_no_int, grid_with_int) %>%
  ggplot(aes(x = x1, y = x2, fill = prediction)) +
  geom_tile() +
  scale_fill_viridis_c() +
  facet_wrap(~model) +
  labs(
    title = "Effect of Including Interaction Terms",
    subtitle = "Interaction allows the effect of x1 to depend on x2",
    x = "Feature 1",
    y = "Feature 2",
    fill = "Predicted\nValue"
  )

# Compare model performance
tibble(
  Model = c("Without Interaction", "With Interaction"),
  RMSE = c(
    sqrt(mean((interaction_data$y - predict(no_int_fit, interaction_data)$.pred)^2)),
    sqrt(mean((interaction_data$y - predict(with_int_fit, interaction_data)$.pred)^2))
  )
) %>%
  knitr::kable(digits = 3)

# Create data with different missing patterns
set.seed(123)
missing_data <- tibble(
  x1 = rnorm(1000),
  x2 = rnorm(1000),
  x3 = x1 + x2 + rnorm(1000, 0, 0.5),
  y = 2*x1 + 3*x2 + x3 + rnorm(1000)
)

# Introduce different missing patterns
missing_data <- missing_data %>%
  mutate(
    # MCAR: Random 20% missing
    x1_mcar = ifelse(runif(n()) < 0.2, NA, x1),
    # MAR: Missing depends on x2
    x2_mar = ifelse(x2 < quantile(x2, 0.2), NA, x2),
    # MNAR: Large values more likely missing
    x3_mnar = ifelse(x3 > quantile(x3, 0.8) & runif(n()) < 0.5, NA, x3)
  )

# Different imputation strategies
imputation_recipes <- list(
  # Remove rows with missing data
  complete_case = recipe(y ~ x1_mcar + x2_mar + x3_mnar, data = missing_data) %>%
    step_naomit(all_predictors()),
  
  # Mean imputation
  mean_imp = recipe(y ~ x1_mcar + x2_mar + x3_mnar, data = missing_data) %>%
    step_impute_mean(all_predictors()),
  
  # Median imputation (robust to outliers)
  median_imp = recipe(y ~ x1_mcar + x2_mar + x3_mnar, data = missing_data) %>%
    step_impute_median(all_predictors()),
  
  # K-nearest neighbors imputation
  knn_imp = recipe(y ~ x1_mcar + x2_mar + x3_mnar, data = missing_data) %>%
    step_impute_knn(all_predictors(), neighbors = 5),
  
  # Linear imputation (using other variables)
  linear_imp = recipe(y ~ x1_mcar + x2_mar + x3_mnar, data = missing_data) %>%
    step_impute_linear(x1_mcar, impute_with = imp_vars(x2_mar, x3_mnar)) %>%
    step_impute_linear(x2_mar, impute_with = imp_vars(x1_mcar, x3_mnar)) %>%
    step_impute_linear(x3_mnar, impute_with = imp_vars(x1_mcar, x2_mar))
)

# Apply imputation and evaluate
imputation_results <- map_df(names(imputation_recipes), function(name) {
  imputed <- imputation_recipes[[name]] %>%
    prep() %>%
    bake(new_data = NULL)
  
  # Calculate statistics
  tibble(
    method = name,
    n_rows = nrow(imputed),
    x1_mean_error = mean(imputed$x1_mcar - missing_data$x1[!is.na(imputed$x1_mcar)], 
                         na.rm = TRUE),
    x2_mean_error = mean(imputed$x2_mar - missing_data$x2[!is.na(imputed$x2_mar)], 
                         na.rm = TRUE),
    x3_mean_error = mean(imputed$x3_mnar - missing_data$x3[!is.na(imputed$x3_mnar)], 
                         na.rm = TRUE)
  )
})

# Visualize imputation quality
imputation_results %>%
  pivot_longer(cols = contains("error"), 
               names_to = "variable", 
               values_to = "error") %>%
  ggplot(aes(x = method, y = abs(error), fill = variable)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d() +
  labs(
    title = "Imputation Error by Method",
    subtitle = "Lower is better - comparing imputed values to true values",
    x = "Imputation Method",
    y = "Absolute Mean Error"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Show data loss with complete case
tibble(
  Method = c("Complete Case", "Imputation Methods"),
  `Rows Retained` = c(
    imputation_results %>% filter(method == "complete_case") %>% pull(n_rows),
    1000
  ),
  `Percentage` = c(
    imputation_results %>% filter(method == "complete_case") %>% pull(n_rows) / 10,
    100
  )
) %>%
  knitr::kable()

# Create data with relevant and irrelevant features
set.seed(123)
feature_data <- tibble(
  # Relevant features
  relevant_1 = rnorm(500),
  relevant_2 = rnorm(500),
  relevant_3 = rnorm(500),
  # Irrelevant features
  noise_1 = rnorm(500),
  noise_2 = rnorm(500),
  noise_3 = rnorm(500),
  noise_4 = rnorm(500),
  # Target depends only on relevant features
  target = 2*relevant_1 + 3*relevant_2 - relevant_3 + rnorm(500, 0, 0.5)
)

# Calculate correlations with target
correlations <- feature_data %>%
  select(-target) %>%
  map_dbl(~ cor(., feature_data$target, use = "complete.obs")) %>%
  enframe(name = "feature", value = "correlation") %>%
  mutate(
    abs_correlation = abs(correlation),
    feature_type = ifelse(str_detect(feature, "relevant"), "Relevant", "Noise")
  )

# Visualize correlations
ggplot(correlations, aes(x = reorder(feature, abs_correlation), 
                         y = abs_correlation, 
                         fill = feature_type)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Relevant" = "darkgreen", "Noise" = "gray50")) +
  geom_hline(yintercept = 0.1, linetype = "dashed", color = "red") +
  labs(
    title = "Feature Selection Using Correlation Filter",
    subtitle = "Red line shows potential threshold for feature selection",
    x = "Feature",
    y = "Absolute Correlation with Target",
    fill = "True Feature Type"
  )

# Recipe with correlation filter
filtered_recipe <- recipe(target ~ ., data = feature_data) %>%
  step_corr(all_predictors(), threshold = 0.9) %>%  # Remove highly correlated features
  step_rm(all_predictors(), -all_outcomes(), 
          skip = FALSE,
          threshold = 0.1)  # This would remove low correlation features

# Near-zero variance filter
nzv_recipe <- recipe(target ~ ., data = feature_data) %>%
  step_nzv(all_predictors())  # Remove features with near-zero variance

# Create correlated features for PCA demonstration
pca_data <- tibble(
  x1 = rnorm(500),
  x2 = x1 + rnorm(500, 0, 0.5),  # Correlated with x1
  x3 = rnorm(500),
  x4 = x3 + rnorm(500, 0, 0.5),  # Correlated with x3
  x5 = rnorm(500),
  y = x1 + x3 + rnorm(500, 0, 0.5)
)

# PCA recipe
pca_recipe <- recipe(y ~ ., data = pca_data) %>%
  step_normalize(all_predictors()) %>%  # Important: normalize before PCA
  step_pca(all_predictors(), num_comp = 3)  # Keep 3 components

# Prepare and examine
pca_prep <- prep(pca_recipe)
pca_result <- bake(pca_prep, new_data = NULL)

# Extract loadings
pca_loadings <- tidy(pca_prep, 2) %>%  # 2nd step is PCA
  filter(component %in% paste0("PC", 1:3)) %>%
  mutate(
    component = factor(component, levels = paste0("PC", 1:3))
  )

# Visualize loadings
ggplot(pca_loadings, aes(x = terms, y = value, fill = value > 0)) +
  geom_col() +
  facet_wrap(~component) +
  coord_flip() +
  scale_fill_manual(values = c("FALSE" = "red", "TRUE" = "blue")) +
  labs(
    title = "PCA Loadings",
    subtitle = "How original features contribute to each principal component",
    x = "Original Feature",
    y = "Loading",
    fill = "Sign"
  ) +
  theme(legend.position = "none")

# Variance explained
pca_variance <- pca_prep$steps[[2]]$res$sdev^2
variance_explained <- tibble(
  PC = paste0("PC", 1:length(pca_variance)),
  Variance = pca_variance,
  `Proportion Explained` = Variance / sum(Variance),
  `Cumulative Proportion` = cumsum(`Proportion Explained`)
)

# Scree plot
ggplot(variance_explained %>% head(5), 
       aes(x = PC, y = `Proportion Explained`)) +
  geom_col(fill = "steelblue") +
  geom_line(aes(group = 1), color = "red", linewidth = 1) +
  geom_point(size = 3, color = "red") +
  geom_text(aes(label = round(`Cumulative Proportion`, 2)), 
            vjust = -1, size = 3) +
  labs(
    title = "PCA Scree Plot",
    subtitle = "Shows variance explained by each component",
    x = "Principal Component",
    y = "Proportion of Variance Explained"
  )

# Create time series data
time_data <- tibble(
  date = seq(as.Date("2020-01-01"), as.Date("2022-12-31"), by = "day"),
  base_value = 1000,
  trend = seq(0, 200, length.out = length(date)),
  seasonal = 100 * sin(2 * pi * as.numeric(date) / 365),
  weekly = 50 * sin(2 * pi * wday(date) / 7),
  noise = rnorm(length(date), 0, 30),
  sales = base_value + trend + seasonal + weekly + noise
)

# Time-based feature engineering
time_features <- time_data %>%
  mutate(
    # Basic time components
    year = year(date),
    month = month(date),
    day = day(date),
    day_of_week = wday(date, label = TRUE),
    day_of_year = yday(date),
    week_of_year = week(date),
    quarter = quarter(date),
    
    # Cyclical encoding (preserves continuity)
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12),
    day_sin = sin(2 * pi * day / 31),
    day_cos = cos(2 * pi * day / 31),
    
    # Binary indicators
    is_weekend = wday(date) %in% c(1, 7),
    is_month_start = day <= 7,
    is_month_end = day >= day(ceiling_date(date, "month") - days(7)),
    
    # Lag features
    sales_lag_1 = lag(sales, 1),
    sales_lag_7 = lag(sales, 7),
    sales_lag_30 = lag(sales, 30),
    
    # Rolling statistics
    sales_ma_7 = zoo::rollmean(sales, 7, fill = NA, align = "right"),
    sales_ma_30 = zoo::rollmean(sales, 30, fill = NA, align = "right"),
    sales_std_7 = zoo::rollapply(sales, 7, sd, fill = NA, align = "right")
  )

# Visualize some engineered features
feature_importance <- time_features %>%
  drop_na() %>%
  select(sales, month_sin, month_cos, is_weekend, sales_lag_7, sales_ma_30) %>%
  cor() %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  filter(feature != "sales") %>%
  select(feature, correlation = sales) %>%
  arrange(desc(abs(correlation)))

ggplot(feature_importance, aes(x = reorder(feature, abs(correlation)), 
                               y = abs(correlation))) +
  geom_col(fill = "darkblue") +
  coord_flip() +
  labs(
    title = "Importance of Time-Based Features",
    subtitle = "Correlation with sales",
    x = "Feature",
    y = "Absolute Correlation"
  )

# Show cyclical encoding
cyclical_demo <- time_features %>%
  select(month, month_sin, month_cos) %>%
  distinct() %>%
  arrange(month)

ggplot(cyclical_demo, aes(x = month_cos, y = month_sin)) +
  geom_path(color = "blue", linewidth = 1) +
  geom_point(aes(color = factor(month)), size = 3) +
  geom_text(aes(label = month), vjust = -1) +
  coord_equal() +
  scale_color_viridis_d() +
  labs(
    title = "Cyclical Encoding of Months",
    subtitle = "Preserves continuity: December is close to January",
    x = "Cosine Component",
    y = "Sine Component"
  ) +
  theme(legend.position = "none")

# Simple text feature engineering example
text_data <- tibble(
  id = 1:5,
  review = c(
    "This product is absolutely amazing! Best purchase ever!",
    "Terrible quality. Very disappointed. Would not recommend.",
    "Good value for money. Satisfied with the purchase.",
    "Excellent service and fast delivery. Five stars!",
    "Product broke after one day. Complete waste of money."
  ),
  rating = c(5, 1, 4, 5, 1)
)

# Text feature recipe using textrecipes
text_recipe <- recipe(rating ~ review, data = text_data) %>%
  # Tokenize text
  step_tokenize(review) %>%
  # Remove stop words
  step_stopwords(review) %>%
  # Create n-grams
  step_ngram(review, num_tokens = 2) %>%
  # Convert to term frequency
  step_tf(review, weight_scheme = "binary") %>%
  # Optionally: TF-IDF weighting
  # step_tfidf(review) %>%
  # Keep only most common terms
  step_tokenfilter(review, max_tokens = 20)

# Note: This is just a demonstration - real text processing needs more data

# Create split if not already done
if (!exists("ames_split")) {
  set.seed(123)
  ames_split <- initial_split(ames, prop = 0.75, strata = Sale_Price)
  ames_train <- training(ames_split)
  ames_test <- testing(ames_split)
}

# WRONG: Normalizing before splitting
# This leaks information from test set into training
wrong_approach <- ames %>%
  mutate(Gr_Liv_Area_scaled = scale(Gr_Liv_Area)[,1]) %>%  # Uses ALL data!
  initial_split()

# RIGHT: Normalize within recipe
right_recipe <- recipe(Sale_Price ~ Gr_Liv_Area, data = training(ames_split)) %>%
  step_normalize(Gr_Liv_Area)  # Will use only training data statistics

# WRONG: Creating too many features
overengineered_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_poly(all_numeric_predictors(), degree = 5) %>%  # Too many polynomial terms
  step_interact(terms = ~ all_numeric_predictors()^2)  # All 2-way interactions

# RIGHT: Thoughtful feature engineering
thoughtful_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_log(Sale_Price) %>%
  step_poly(Gr_Liv_Area, degree = 2) %>%  # Only where needed
  step_interact(terms = ~ Gr_Liv_Area:Overall_Cond)  # Specific, meaningful interaction

# Use credit data for a complete example
credit_split <- initial_split(credit_data, prop = 0.75, strata = Status)
credit_train <- training(credit_split)
credit_test <- testing(credit_split)

# Comprehensive recipe
comprehensive_recipe <- recipe(Status ~ ., data = credit_train) %>%
  # 1. Handle missing values
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  
  # 2. Feature creation
  step_mutate(
    debt_to_income = Expenses / Income,
    savings_rate = (Income - Expenses) / Income,
    has_records = !is.na(Records)
  ) %>%
  
  # 3. Handle skewness
  step_YeoJohnson(Income, Amount) %>%
  
  # 4. Create dummy variables
  step_dummy(all_nominal_predictors(), -has_records) %>%
  
  # 5. Remove near-zero variance
  step_nzv(all_predictors()) %>%
  
  # 6. Normalize
  step_normalize(all_numeric_predictors()) %>%
  
  # 7. Remove highly correlated features
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  
  # 8. PCA for dimensionality reduction (optional)
  # step_pca(all_numeric_predictors(), threshold = 0.95) %>%
  
  # 9. Balance classes (for classification)
  step_smote(Status)  # Synthetic minority oversampling

# Examine the recipe
comprehensive_recipe

# Prepare and check results
comprehensive_prep <- prep(comprehensive_recipe)
comprehensive_baked <- bake(comprehensive_prep, new_data = NULL)

# Summary of transformations
tibble(
  Stage = c("Original", "After Engineering"),
  `N Features` = c(ncol(credit_train) - 1, ncol(comprehensive_baked) - 1),
  `N Observations` = c(nrow(credit_train), nrow(comprehensive_baked)),
  `Class Balance` = c(
    sum(credit_train$Status == "good") / nrow(credit_train),
    sum(comprehensive_baked$Status == "good") / nrow(comprehensive_baked)
  )
) %>%
  knitr::kable(digits = 3)

# Fit a model with the engineered features
rf_spec <- rand_forest(trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("classification")

workflow() %>%
  add_recipe(comprehensive_recipe) %>%
  add_model(rf_spec) %>%
  fit(credit_train) %>%
  predict(credit_test) %>%
  bind_cols(credit_test) %>%
  accuracy(truth = Status, estimate = .pred_class)

# Your solution
exercise_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>%
  # Transform outcome
  step_log(Sale_Price) %>%
  
  # Create meaningful features
  step_mutate(
    House_Age = 2010 - Year_Built,
    Remod_Age = 2010 - Year_Remod_Add,
    Has_Garage = !is.na(Garage_Type),
    Total_Bathrooms = Full_Bath + Half_Bath * 0.5,
    Total_SF = Gr_Liv_Area + Total_Bsmt_SF,
    Quality_x_Condition = Overall_Cond * Overall_Cond
  ) %>%
  
  # Remove original year variables
  step_rm(Year_Built, Year_Remod_Add, Mo_Sold, Year_Sold) %>%
  
  # Handle missing values
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  
  # Rare categories
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  
  # Transform skewed variables (before creating dummies)
  step_YeoJohnson(Lot_Area, Gr_Liv_Area, Total_SF) %>%
  
  # Interactions (before creating dummies)
  step_interact(terms = ~ Total_SF:Overall_Cond) %>%
  
  # Create dummies
  step_dummy(all_nominal_predictors()) %>%
  
  # Scale
  step_normalize(all_numeric_predictors()) %>%
  
  # Remove zero variance
  step_nzv(all_predictors())

# Test the recipe
exercise_prep <- prep(exercise_recipe)
exercise_baked <- bake(exercise_prep, new_data = NULL)

print(paste("Created", ncol(exercise_baked) - 1, "features from", 
            ncol(ames_train) - 1, "original features"))

# Your solution
# Create synthetic high-cardinality data
high_card_ex <- tibble(
  category = sample(paste0("Cat_", 1:100), 1000, replace = TRUE),
  value = rnorm(1000),
  target = rnorm(1000)
) %>%
  mutate(
    # Make target depend somewhat on category
    target = target + as.numeric(factor(category)) / 20
  )

# Try different encoding strategies
strategies <- list(
  # Frequency encoding
  frequency = recipe(target ~ ., data = high_card_ex) %>%
    step_mutate(cat_freq = n(), .by = category) %>%
    step_rm(category),
  
  # Target encoding with smoothing
  target_enc = recipe(target ~ ., data = high_card_ex) %>%
    step_mutate(
      cat_mean = mean(target),
      cat_count = n(),
      .by = category
    ) %>%
    step_mutate(
      # Smooth with global mean for rare categories
      cat_smooth = (cat_mean * cat_count + mean(target) * 10) / (cat_count + 10)
    ) %>%
    step_rm(category, cat_mean, cat_count),
  
  # Embedding-like (PCA on dummies)
  embedding = recipe(target ~ ., data = high_card_ex) %>%
    step_dummy(category) %>%
    step_pca(starts_with("category_"), num_comp = 10)
)

# Compare approaches
comparison <- map_df(names(strategies), function(name) {
  prepped <- prep(strategies[[name]])
  baked <- bake(prepped, new_data = NULL)
  
  tibble(
    method = name,
    n_features = ncol(baked) - 1
  )
})

print(comparison)

# Your solution
# Generate time series data
ts_exercise <- tibble(
  date = seq(as.Date("2021-01-01"), as.Date("2023-12-31"), by = "day")
) %>%
  mutate(
    trend = row_number() / n() * 100,
    seasonal = 50 * sin(2 * pi * yday(date) / 365),
    weekly = 20 * sin(2 * pi * wday(date) / 7),
    noise = rnorm(n(), 0, 10),
    sales = 1000 + trend + seasonal + weekly + noise
  )

# Create time features
ts_features <- ts_exercise %>%
  mutate(
    # Calendar features
    year = year(date),
    month = month(date),
    week = week(date),
    day_of_week = wday(date),
    day_of_month = day(date),
    day_of_year = yday(date),
    quarter = quarter(date),
    
    # Cyclical encoding
    month_sin = sin(2 * pi * month / 12),
    month_cos = cos(2 * pi * month / 12),
    dow_sin = sin(2 * pi * day_of_week / 7),
    dow_cos = cos(2 * pi * day_of_week / 7),
    
    # Indicators
    is_weekend = day_of_week %in% c(1, 7),
    is_month_start = day_of_month <= 3,
    is_month_end = day_of_month >= 28,
    
    # Lag features
    sales_lag_1 = lag(sales, 1),
    sales_lag_7 = lag(sales, 7),
    sales_lag_30 = lag(sales, 30),
    sales_lag_365 = lag(sales, 365),
    
    # Rolling statistics
    sales_ma_7 = zoo::rollmean(sales, 7, fill = NA, align = "right"),
    sales_ma_30 = zoo::rollmean(sales, 30, fill = NA, align = "right"),
    sales_std_7 = zoo::rollapply(sales, 7, sd, fill = NA, align = "right"),
    sales_std_30 = zoo::rollapply(sales, 30, sd, fill = NA, align = "right"),
    
    # Differences
    sales_diff_1 = sales - lag(sales, 1),
    sales_diff_7 = sales - lag(sales, 7)
  ) %>%
  drop_na()

# Evaluate feature importance
feature_cors <- ts_features %>%
  select(-date, -trend, -seasonal, -weekly, -noise) %>%
  select(-sales) %>%
  map_dbl(~ cor(., ts_features$sales)) %>%
  enframe(name = "feature", value = "correlation") %>%
  arrange(desc(abs(correlation))) %>%
  head(15)

ggplot(feature_cors, aes(x = reorder(feature, abs(correlation)), 
                         y = abs(correlation))) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top Time Series Features",
    subtitle = "Absolute correlation with sales",
    x = "Feature",
    y = "|Correlation|"
  )
