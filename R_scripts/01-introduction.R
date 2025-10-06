# Install tidyverse if you haven't already
# install.packages("tidyverse")

# Load the tidyverse
library(tidyverse)

# Traditional nested approach (hard to read)
# We read this from inside-out: first sqrt, then mean, then round
# But we write it outside-in!
round(mean(sqrt(c(1, 4, 9, 16, 25))), 2)

# With pipes (much clearer!)
# We read AND write this in the order operations happen
c(1, 4, 9, 16, 25) %>%
  sqrt() %>%         # First: take square root
  mean() %>%         # Then: calculate mean
  round(2)           # Finally: round to 2 decimals

# Using the native R pipe (R 4.1+)
# The |> operator is built into base R as of version 4.1
# It works similarly to %>% but with some subtle differences
c(1, 4, 9, 16, 25) |>
  sqrt() |>
  mean() |>
  round(2)

# Using the dot placeholder for custom positioning
10 %>% 
  `/`(2) %>%      # 10 / 2 = 5
  `+`(3) %>%      # 5 + 3 = 8  
  `^`(2)          # 8^2 = 64

# When the piped value isn't the first argument
5 %>% 
  seq(from = 1, to = .)  # Creates sequence from 1 to 5

# More practical example with data
mtcars %>%
  lm(mpg ~ cyl + wt, data = .)  # data = . puts the piped data in the right place

# Load a built-in dataset
data(mtcars)

# Without pipes - nested and hard to read
head(arrange(filter(mtcars, cyl == 6), desc(mpg)), 5)

# With pipes - clear and sequential
mtcars %>%
  filter(cyl == 6) %>%
  arrange(desc(mpg)) %>%
  head(5)

# Create a tibble from scratch
my_tibble <- tibble(
  name = c("Alice", "Bob", "Charlie", "Diana"),
  age = c(25, 30, 35, 28),
  score = c(85.5, 92.3, 78.9, 88.1),
  passed = c(TRUE, TRUE, FALSE, TRUE)
)

my_tibble

# Convert data frame to tibble
mtcars_tibble <- as_tibble(mtcars)

# Compare printing
print("Data frame (first 6 rows shown by default):")
head(mtcars)

print("Tibble (shows what fits on screen):")
mtcars_tibble

# Tibbles preserve data types
df <- data.frame(x = 1:3, y = c("a", "b", "c"))
tb <- tibble(x = 1:3, y = c("a", "b", "c"))

# Data frame converts strings to factors (in older R versions)
str(df)
str(tb)

# Tibbles handle column names better
weird_tb <- tibble(
  `First Name` = c("John", "Jane"),
  `2020` = c(100, 200),
  `:)` = c("happy", "sad")
)
weird_tb

# Messy data
messy_data <- tibble(
  student = c("Alice", "Bob", "Charlie"),
  midterm = c(85, 90, 78),
  final = c(88, 85, 92)
)

print("Messy data (wide format):")
messy_data

# Tidy data
tidy_data <- messy_data %>%
  pivot_longer(
    cols = c(midterm, final),
    names_to = "exam",
    values_to = "score"
  )

print("Tidy data (long format):")
tidy_data

# Load the palmerpenguins package for example data
# install.packages("palmerpenguins")
library(palmerpenguins)

# Explore the penguins dataset
glimpse(penguins)

# Basic exploration
penguins %>%
  summary()

# Complete analysis pipeline
penguins %>%
  drop_na() %>%  # Remove missing values
  group_by(species, island) %>%
  summarise(
    count = n(),
    avg_bill_length = mean(bill_length_mm),
    avg_body_mass = mean(body_mass_g),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_body_mass))

penguins %>%
  drop_na() %>%
  ggplot(aes(x = flipper_length_mm, y = body_mass_g, color = species)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Penguin Body Mass vs Flipper Length",
    x = "Flipper Length (mm)",
    y = "Body Mass (g)",
    color = "Species"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")

# Typical workflow
penguins %>%
  # Clean
  drop_na() %>%
  filter(year == 2008) %>%
  # Transform
  mutate(
    body_mass_kg = body_mass_g / 1000,
    size_category = case_when(
      body_mass_kg < 3.5 ~ "Small",
      body_mass_kg < 4.5 ~ "Medium",
      TRUE ~ "Large"
    )
  ) %>%
  # Summarize
  group_by(species, size_category) %>%
  summarise(count = n(), .groups = "drop") %>%
  # Visualize
  ggplot(aes(x = species, y = count, fill = size_category)) +
  geom_col(position = "dodge") +
  theme_minimal() +
  labs(title = "Penguin Size Distribution by Species (2008)")

# Complex grouped operations
penguins %>%
  drop_na() %>%
  group_by(species) %>%
  mutate(
    bill_length_z = (bill_length_mm - mean(bill_length_mm)) / sd(bill_length_mm),
    bill_depth_z = (bill_depth_mm - mean(bill_depth_mm)) / sd(bill_depth_mm)
  ) %>%
  filter(abs(bill_length_z) < 2 & abs(bill_depth_z) < 2) %>%  # Remove outliers
  summarise(
    n = n(),
    correlation = cor(bill_length_mm, bill_depth_mm),
    .groups = "drop"
  )

# Your code here
1:100 %>%
  keep(~ . %% 2 == 0) %>%  # Keep even numbers
  map_dbl(~ .^2) %>%        # Square each
  mean() %>%                # Calculate mean
  sqrt()                    # Take square root

# Your code here
books <- tibble(
  title = c("The Great Gatsby", "1984", "The Hunger Games", "Dune", "Project Hail Mary"),
  author = c("F. Scott Fitzgerald", "George Orwell", "Suzanne Collins", "Frank Herbert", "Andy Weir"),
  year = c(1925, 1949, 2008, 1965, 2021),
  pages = c(180, 328, 374, 688, 476),
  rating = c(4.5, 4.8, 4.3, 4.7, 4.9)
)

books %>%
  filter(year > 2000) %>%
  mutate(reading_time_hours = pages / 60) %>%
  arrange(desc(rating))

# Your code here
# Part 1 & 2
penguins %>%
  drop_na(bill_length_mm) %>%
  group_by(species, island) %>%
  summarise(
    avg_bill_length = mean(bill_length_mm),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_bill_length))

# Part 3
penguins %>%
  drop_na(body_mass_g) %>%
  group_by(species) %>%
  summarise(
    min_mass = min(body_mass_g),
    mean_mass = mean(body_mass_g),
    max_mass = max(body_mass_g),
    .groups = "drop"
  )

# Given data
student_scores <- tibble(
  student_id = 1:5,
  math_score = c(85, 92, 78, 95, 88),
  science_score = c(90, 88, 85, 92, 91),
  english_score = c(88, 85, 90, 87, 89)
)

# Your code here
student_scores %>%
  pivot_longer(
    cols = ends_with("_score"),
    names_to = "subject",
    values_to = "score",
    names_pattern = "(.*)_score"
  ) %>%
  group_by(subject) %>%
  summarise(
    avg_score = mean(score),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_score))
