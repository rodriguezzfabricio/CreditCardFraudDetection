# Load the dataset
credit_card <- read.csv('/Users/fabriciorodriguez/Downloads/archive/fraudTrain.csv')
View(credit_card)

# Convert is_fraud column to factor
credit_card$is_fraud <- factor(credit_card$is_fraud, levels = c(0, 1))

# Summary of the dataset
summary(credit_card)

# Column names of the dataset
colnames(credit_card)

# Frequency of fraud and non-fraud cases
table(credit_card$is_fraud)

# Check for missing values
sum(is.na(credit_card))

# Proportion of fraud and non-fraud cases
prop.table(table(credit_card$is_fraud))

# Labels for pie chart
checker <- c("legit", "fraud")
checker <- paste(checker, round(100 * prop.table(table(credit_card$is_fraud)), 2))
checker <- paste0(checker, "%")

# Pie chart for fraud vs legit transactions
pie(table(credit_card$is_fraud), checker, col = c("blue", "red"), 
    main = "Pie Chart Representing Fraud Cases")

# Baseline predictions vector
predictions <- rep.int(0, nrow(credit_card))
predictions <- factor(predictions, levels = c(0, 1))

# Confusion Matrix using caret package
library(caret)
confusionMatrix(data = predictions, reference = credit_card$is_fraud)

# Load dplyr package
library(dplyr)

# Sample 10% of the data for visualization
set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)

# Verify fraud proportions in the sample
table(credit_card$is_fraud)

# Load ggplot2 package
library(ggplot2)

# Boxplot: Transaction Amount by Fraud Type
ggplot(data = credit_card, aes(x = is_fraud, y = amt, fill = is_fraud)) +
  geom_boxplot() +
  labs(title = "Transaction Amount by Fraud Type", x = "Fraud Type", y = "Transaction Amount") +
  scale_fill_manual(values = c("blue", "red"), labels = c("Legit", "Fraud"))

# Scatter Plot: Transaction Amount vs City Population
ggplot(data = credit_card, aes(x = amt, y = city_pop, col = is_fraud)) +
  geom_point(alpha = 0.6) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'), labels = c("Legit", "Fraud")) +
  labs(title = "Scatter Plot of Transaction Amount vs City Population",
       x = "Transaction Amount",
       y = "City Population",
       color = "Fraud Type")
