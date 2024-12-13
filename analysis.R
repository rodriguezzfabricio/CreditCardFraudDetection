# ------------------- Install and Load Necessary Libraries -------------------

# Uncomment to install required packages
# install.packages("caret")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("smotefamily")
# install.packages("ROSE")

library(caret)
library(ggplot2)
library(dplyr)
library(rpart)
library(rpart.plot)
library(smotefamily)
library(ROSE)

# ------------------- Load and Preprocess the Dataset -------------------

# Load the dataset
credit_card <- read.csv('/Users/fabriciorodriguez/Downloads/archive/fraudTrain.csv')
test_data <- read.csv('/Users/fabriciorodriguez/Downloads/archive/fraudTest.csv')

# Convert 'is_fraud' to a factor
credit_card$is_fraud <- factor(credit_card$is_fraud, levels = c(0, 1))
test_data$is_fraud <- factor(test_data$is_fraud, levels = c(0, 1))

# View summary statistics
summary(credit_card)
table(credit_card$is_fraud)
prop.table(table(credit_card$is_fraud))

# ------------------- Data Visualization -------------------

# Visualize the original dataset
ggplot(data = credit_card, aes(x = amt, y = city_pop, col = is_fraud)) +
  geom_point(alpha = 0.6) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red')) +
  labs(title = "Scatter Plot of Transaction Amount vs City Population",
       x = "Transaction Amount",
       y = "City Population",
       color = "Fraud Type")

# ------------------- SMOTE Implementation -------------------

# Setting parameters for SMOTE
n0 <- table(credit_card$is_fraud)[1] # Count of majority class
n1 <- table(credit_card$is_fraud)[2] # Count of minority class
r0 <- 0.6   # Desired proportion of majority class

# Calculate duplication size for SMOTE
ntimes <- ((1 - r0) / r0) * (n0 / n1) - 1

# Apply SMOTE
smote_output <- SMOTE(X = credit_card[, -which(names(credit_card) %in% c("is_fraud"))], 
                      target = credit_card$is_fraud, 
                      K = 5, 
                      dup_size = ntimes)

# Extract the oversampled dataset
credit_smote <- smote_output$data
colnames(credit_smote)[ncol(credit_smote)] <- "is_fraud"
credit_smote$is_fraud <- factor(credit_smote$is_fraud, levels = c(0, 1))

# Verify the class distribution after SMOTE
prop.table(table(credit_smote$is_fraud))

# ------------------- Random Oversampling (ROS) and Undersampling (RUS) -------------------

# Random Over Sampling (ROS)
oversampling_result <- ovun.sample(is_fraud ~ ., 
                                   data = credit_card, 
                                   method = "over", 
                                   N = nrow(credit_card) * 2, 
                                   seed = 2019)
oversampled_credit <- oversampling_result$data

# Random Under Sampling (RUS)
undersampling_result <- ovun.sample(is_fraud ~ ., 
                                    data = credit_card, 
                                    method = "under", 
                                    N = 70, 
                                    seed = 2019)
undersampled_credit <- undersampling_result$data

# Combination of ROS and RUS
combined_sampling_result <- ovun.sample(is_fraud ~ ., 
                                        data = credit_card, 
                                        method = "both", 
                                        N = nrow(credit_card), 
                                        p = 0.5, 
                                        seed = 2019)
combined_credit <- combined_sampling_result$data

# ------------------- Decision Tree Models -------------------

# Build a decision tree model using SMOTE dataset
CART_model_smote <- rpart(is_fraud ~ ., 
                          data = credit_smote, 
                          method = "class")

# Plot the decision tree for SMOTE
rpart.plot(CART_model_smote, 
           extra = 0, 
           type = 5, 
           tweak = 1.2, 
           main = "Decision Tree with SMOTE")

# Predict on test data using SMOTE model
predicted_smote <- predict(CART_model_smote, test_data[, -which(names(test_data) %in% c("is_fraud"))], type = "class")

# Build a confusion matrix for SMOTE model
confusionMatrix(predicted_smote, test_data$is_fraud)

# Build a decision tree model using the original dataset
CART_model_original <- rpart(is_fraud ~ ., 
                             data = credit_card, 
                             method = "class")

# Plot the decision tree for the original dataset
rpart.plot(CART_model_original, 
           extra = 0, 
           type = 5, 
           tweak = 1.2, 
           main = "Decision Tree without SMOTE")

# Predict on test data using the original model
predicted_original <- predict(CART_model_original, test_data[, -which(names(test_data) %in% c("is_fraud"))], type = "class")

# Build a confusion matrix for the original model
confusionMatrix(predicted_original, test_data$is_fraud)

# ------------------- Performance Comparison -------------------

cat("\nPerformance of SMOTE-based Model:\n")
print(confusionMatrix(predicted_smote, test_data$is_fraud))

cat("\nPerformance of Original Dataset Model:\n")
print(confusionMatrix(predicted_original, test_data$is_fraud))

# ------------------- Visualization for SMOTE -------------------

ggplot(credit_smote, aes(x = amt, y = city_pop, color = is_fraud)) +
  geom_point(alpha = 0.6) +
  theme_bw() +
  scale_color_manual(values = c("dodgerblue2", "red")) +
  labs(title = "SMOTE - Oversampled Data",
       x = "Transaction Amount",
       y = "City Population",
       color = "Fraud Type")
