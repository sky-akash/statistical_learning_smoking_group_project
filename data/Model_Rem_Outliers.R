library(readr)
Smoking_Training_Data <- read_csv("Health Dataset/Kaggle_Smoking/Smoking_Training_Data.csv")

#Loading Libraries
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(pscl)
library(car)
library(corrplot)

#Load Dataset
attach(Smoking_Training_Data)

#Create Dataframe
df = Smoking_Training_Data

# Head-Sstructure and Summary along with Null Check 
head(df)
str(df)
colnames(df)
##
df <- na.omit(df)
##
str(df)
summary(df)

## So No NUll Values, Good!
# Renaming the column Names for convenience

df <- df %>%
  rename(
    height_cm = `height(cm)`,
    weight_kg = `weight(kg)`,
    waist_cm = `waist(cm)`,
    eyesight_left = `eyesight(left)`,
    eyesight_right = `eyesight(right)`,
    hearing_left = `hearing(left)`,
    hearing_right = `hearing(right)`,
    fasting_blood_sugar = `fasting blood sugar`,
    urine_protein = `Urine protein`,
    serum_creatinine = `serum creatinine`,
    dental_caries = `dental caries`
  )

# Factoring the Factor Variables
df <-df %>%
  mutate(
    hearing_left = factor(hearing_left),
    hearing_right = factor(hearing_right),
    dental_caries = factor(dental_caries),
    smoking = factor(smoking)
  )

#Adding Extra Columns to dataset for Derived Variables 
# Var1 is WHtR (Waist to Height Ratio) = waist (cm)/ height (cm)
# Var2 is as_lt i.e. AST / ALT ratio
# Var3 is BMI, Body Mass Index = Weight (kg)/ height^2 (m)

# df_fe -> Feature Engineered Dataframe

df_fe_unclean <- df %>%
  mutate(
    WHtR = waist_cm / height_cm,
    as_lt = AST / ALT,
    BMI = weight_kg / (height_cm / 100),
    MAP = (systolic + 2 * relaxation) / 3,
    eyesight_avg = (eyesight_left + eyesight_right) / 2
  )

summary(df_fe_unclean)
# Removing Outliers

df_fe <- df_fe_unclean[!(df_fe_unclean$Cholesterol > 240 |
                         df_fe_unclean$LDL > 190 |
                         df_fe_unclean$eyesight_left > 2 |
                         df_fe_unclean$eyesight_right > 2 |
                         df_fe_unclean$fasting_blood_sugar > 126 |
                         df_fe_unclean$systolic > 160 |
                         df_fe_unclean$relaxation > 100 |
                         df_fe_unclean$triglyceride > 200 |
                         df_fe_unclean$HDL < 30 | df_fe_unclean$HDL > 120 |
                         df_fe_unclean$LDL > 200 |
                         df_fe_unclean$hemoglobin > 19 |
                         df_fe_unclean$serum_creatinine > 1.6 |
                         df_fe_unclean$AST < 9 | df_fe_unclean$AST > 60 |
                         df_fe_unclean$ALT < 6 | df_fe_unclean$ALT > 60 |
                         df_fe_unclean$Gtp > 180), ]
summary(df_fe)
str(df_fe)

# Split numeric data based on smoking status
numeric_data_smoker <- df_fe[df_fe$smoking == "1", ]
numeric_data_non_smoker <- df_fe[df_fe$smoking == "0", ]

par(mfrow = c(4, ncol(df_fe) / 4), mar = c(4, 4, 2, 2))
for (i in 1:ncol(df_fe)) {
  boxplot(numeric_data_smoker[, i], col = "skyblue", border = "blue", 
          main = paste("Boxplot of", colnames(df_fe)[i], "for Smokers"))
  boxplot(numeric_data_non_smoker[, i], col = "lightpink", border = "maroon", 
          main = paste("Boxplot of", colnames(df_fe)[i], "for Non-Smokers"))
}


# Splitting Dataset for Training, Validation and Test
set.seed(1)

#To split the dataset randomly into the three categories, let's generate random indices to split the dataset

# Define proportions for splitting
train_prop <- 0.7  # 70% for training
val_prop <- 0.15   # 15% for validation
test_prop <- 0.15  # 15% for testing

# Calculate the number of samples for each set
train_size <- round(train_prop * nrow(df_fe))
val_size <- round(val_prop * nrow(df_fe))
test_size <- nrow(df_fe) - train_size - val_size

indices <- sample(nrow(df_fe))

# Split the data
train_data <- df_fe[indices[1:train_size], ]
val_data <- df_fe[indices[(train_size + 1):(train_size + val_size)], ]
test_data <- df_fe[indices[(train_size + val_size + 1):nrow(df_fe)], ]

# Let's print the dimensions of the split datasets
dim(train_data)
dim(val_data)
dim(test_data)

# Print the Structure of the dataframe
str(df_fe)
summary(df_fe)
# Print the Structure of the training data
str(train_data)
# Summary of Training Data
summary(train_data)

### Note the summary shows there are comparatively More Non-Smokers than Smokers ~ 17k vs 10k
### But for now, we are proceeding with the data as it is without re-sampling of the dataset. Also, it also shows a scenario of Real World population as there is no 50-50 ratio of Smokers to Non-Smokers in Real Population, and there are generally more Non-Smokers.


# Create a layout for the plots
par(mfrow = c(5, 6))

# Iterate over each column in the dataframe
for (col in names(train_data)) {
  if (is.numeric(train_data[[col]])) {
    # Plot histogram for numeric columns
    hist(train_data[[col]], main = paste("Histogram of", col),
         xlab = col, ylab = "Frequency")
  } else {
    # Plot bar plot for factor columns
    barplot(table(train_data[[col]]), main = paste("Bar Plot of", col),
            xlab = col, ylab = "Frequency")
  }
}

## Fit the Model on all parameters (except feature engineered vectors)
#____________________________________
## Model 1
#____________________________________
glm.fits1 <- glm(
  smoking ~ . -MAP -BMI -WHtR -as_lt -eyesight_avg,
  data = train_data, family = binomial
)
summary(glm.fits1)

#Applying the fitted Model 1
glm.probs1 <- predict(glm.fits1, type = "response")

contrasts(train_data$smoking)

glm.pred1 <- rep("0", 18731)
glm.pred1[glm.probs1 > .5] = "1"
###
table(glm.pred1, train_data$smoking)

conf_matrix1 <- confusionMatrix(as.factor(glm.pred1), as.factor(train_data$smoking))
conf_matrix1
f1_score1 <- 2 * (conf_matrix1$byClass['Precision'] * conf_matrix1$byClass['Sensitivity']) / (conf_matrix1$byClass['Precision'] + conf_matrix1$byClass['Sensitivity'])
f1_score1
# ROC curve and AUC
roc_curve1 <- roc(train_data$smoking, glm.probs1) # ROC curve
plot(roc_curve1, main = "ROC Curve 1", col = "blue", lwd = 2) # Plot ROC curve
auc <- auc(roc_curve1) # AUC (Area Under the Curve)
legend("bottomright", legend = paste("AUC =", round(auc, 3)), col = "blue", lwd = 2) # Add AUC to plot

vif_values1 <- vif(glm.fits1)
print(vif_values1)


glm1_coef_names <- names(glm.fits1$coefficients)[-1]
existing_cols1 <- intersect(glm1_coef_names, colnames(train_data))
cor_matrix1 <- cor(train_data[, existing_cols1])

corrplot(cor_matrix1, method = "color", type = "lower", order = "hclust", tl.col = "black", tl.srt = 45, main = "Correlation Plot Initial")

# Another model with less variables
#____________________________________
## Model 2
# Note, here we still have not performed any standardization or normalization on the dataset.
#____________________________________

str(train_data)
numeric_vars_train_data <- train_data[, sapply(train_data, is.numeric)]
cor_matrix_train_data <- cor(numeric_vars_train_data)
corrplot(cor_matrix_train_data, method = "color", type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 45, main = "Correlation Matrix of Numeric Variables in Training Data")

glm.fits2 <- glm(
  smoking ~ .-height_cm -weight_kg -waist_cm -eyesight_left -eyesight_right -systolic -relaxation -LDL -AST -ALT -BMI , data = train_data, family = binomial
)
summary(glm.fits2)

vif_values2 <- vif(glm.fits2)
print(vif_values2)

glm_coef_names2 <- names(glm.fits2$coefficients)[-1]
existing_cols2 <- intersect(glm_coef_names2, colnames(train_data))
cor_matrix2 <- cor(train_data[, existing_cols2])

corrplot(cor_matrix2, method = "color", type = "lower", order = "hclust", tl.col = "black", tl.srt = 45)

# Let us also drop hearing and eyesight variables, (Note* - We also checked their scatter plot and boxplots for any relation with the target vareiable.)
## Now Let's Fit the Different Models and Analyze their Confusion Matrices and f1_score
## Models to Fit - 
# 1. glm.fits1
# 2. glm.fits2 
######
##########################################################################################################################
# 1. glm.fits2

#Applying the fitted glm.fits2 Model
glm.probs2 <- predict(glm.fits2, type = "response")

contrasts(train_data$smoking)

glm.pred2 <- rep("0", 18731)
glm.pred2[glm.probs2 > .5] = "1"
###
table(glm.pred2, train_data$smoking)

conf_matrix2 <- confusionMatrix(as.factor(glm.pred2), as.factor(train_data$smoking))
conf_matrix2

f1_score2 <- 2 * (conf_matrix2$byClass['Precision'] * conf_matrix2$byClass['Sensitivity']) / (conf_matrix2$byClass['Precision'] + conf_matrix2$byClass['Sensitivity'])
f1_score2

# Plot ROC curve
roc_curve_2 <- roc(train_data$smoking, glm.probs2)
plot(roc_curve_2, main="ROC Curve for Model 2", col="yellow4", lwd=2)
abline(a=0, b=1, lty=2, col="red")  # Add diagonal reference line
legend("bottomright", legend=paste("AUC =", round(auc(roc_curve_2), 2)), col="yellow4", lwd=1)
auc_2 <- auc(roc_curve_2)

#_________________________________________________________________________________________________________________________

##########################################################################################################################


## Further, let us apply the fitted models on the Validation Set too.
## Models to test on Validation Set (Be Mindful to apply transformations on Validation Data, as in Training Data)

#
#_____________________________________________________
# Prepare Validation Set for Model 2
# val_data renamed as val_data2 for Model 2 - No need to Do this, as the val_data has all the feature engineered variables (added initially in whole)
###
val_data2 <- val_data
## Fitting the Model on val_data2
# 1. glm.fits2

glm.probs_val2 <- predict(glm.fits2, newdata = val_data2, type = "response")

# Convert predicted probabilities to binary predictions using a threshold of 0.5
glm.pred_val2 <- ifelse(glm.probs_val2 > 0.5, "1", "0")
glm.pred_val2 <- factor(glm.pred_val2, levels = levels(val_data2$smoking))

# Create confusion matrix for validation dataset
conf_matrix_val2 <- confusionMatrix(glm.pred_val2, val_data2$smoking)
conf_matrix_val2

# Calculate F1-score for validation dataset
f1_score_val2 <- 2 * (conf_matrix_val2$byClass['Precision'] * conf_matrix_val2$byClass['Sensitivity']) / 
  (conf_matrix_val2$byClass['Precision'] + conf_matrix_val2$byClass['Sensitivity'])
f1_score_val2

# ROC curve and AUC for validation dataset
roc_curve_val2 <- roc(val_data2$smoking, glm.probs_val2)
plot(roc_curve_val2, main = "ROC Curve for Validation Dataset", col = "pink", lwd = 2)
auc_val2 <- auc(roc_curve_val2)
legend("bottomright", legend = paste("AUC =", round(auc_val2, 3)), col = "blue", lwd = 2)

#____________

# Application on Test Data - Selected Model (Cholesterol)

chosen_model <- glm.fits2

## Predictions on Test Data
glm.probs_test <- predict(chosen_model, newdata = test_data, type = "response")

# Convert predicted probabilities to binary predictions using a threshold of 0.5
glm.pred_test <- ifelse(glm.probs_test > 0.5, "1", "0")
glm.pred_test <- factor(glm.pred_test, levels = levels(test_data$smoking))

# Evaluate Performance
conf_matrix_test <- confusionMatrix(glm.pred_test, test_data$smoking)
f1_score_test <- 2 * (conf_matrix_test$byClass['Precision'] * conf_matrix_test$byClass['Sensitivity']) / 
  (conf_matrix_test$byClass['Precision'] + conf_matrix_test$byClass['Sensitivity'])
roc_curve_test <- roc(test_data$smoking, glm.probs_test)
auc_test <- auc(roc_curve_test)

# Print evaluation metrics
print(conf_matrix_test)
print(f1_score_test)
print(auc_test)

# Plot ROC curve for visual evaluation
plot(roc_curve_test, main = "ROC Curve for Test Dataset", col = "purple", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc_test, 3)), col = "purple", lwd = 2)

###########################################################################################

# Let's Plot the Confusion Matrices of all the fits on a graph

par(mfrow = c(2, 2))


# Confusion Matrix for glm.fits1 on training set
glm.probs_train1 <- predict(glm.fits1, type = "response")
glm.pred_train1 <- ifelse(glm.probs_train1 > 0.5, "1", "0")
glm.pred_train1 <- factor(glm.pred_train1, levels = levels(train_data$smoking))
conf_matrix_train1 <- confusionMatrix(glm.pred_train1, train_data$smoking)
plot(conf_matrix_train1$table, col = c("pink", "lightblue"), main = "Confusion Matrix for Model 1 (Training)", 
     xlab = "Predicted", ylab = "Actual", cex.main = 0.9, cex.axis = 0.8)

# Confusion Matrix for glm.fits2 on training set
glm.probs_train2 <- predict(glm.fits2, type = "response")
glm.pred_train2 <- ifelse(glm.probs_train2 > 0.5, "1", "0")
glm.pred_train2 <- factor(glm.pred_train2, levels = levels(train_data$smoking))
conf_matrix_train2 <- confusionMatrix(glm.pred_train2, train_data$smoking)
plot(conf_matrix_train2$table, col = c("pink", "lightblue"), main = "Confusion Matrix for Model 2 (Training)", 
     xlab = "Predicted", ylab = "Actual", cex.main = 0.9, cex.axis = 0.8)

# Confusion Matrix for glm.fits2 on validation set
glm.probs_val2 <- predict(glm.fits2, newdata = val_data, type = "response")
glm.pred_val2 <- ifelse(glm.probs_val2 > 0.5, "1", "0")
glm.pred_val2 <- factor(glm.pred_val2, levels = levels(val_data$smoking))
conf_matrix_val2 <- confusionMatrix(glm.pred_val2, val_data$smoking)
plot(conf_matrix_val2$table, col = c("pink", "lightblue"), main = "Confusion Matrix for Model 2 (Validation)", 
     xlab = "Predicted", ylab = "Actual", cex.main = 0.9, cex.axis = 0.8)

# Confusion Matrix for glm.fits2 on test set
glm.probs_test2 <- predict(glm.fits2, newdata = test_data, type = "response")
glm.pred_test2 <- ifelse(glm.probs_test2 > 0.5, "1", "0")
glm.pred_test2 <- factor(glm.pred_test2, levels = levels(test_data$smoking))
conf_matrix_test2 <- confusionMatrix(glm.pred_test2, test_data$smoking)
plot(conf_matrix_test2$table, col = c("pink", "lightblue"), main = "Confusion Matrix for Model 2 (Test)", 
     xlab = "Predicted", ylab = "Actual", cex.main = 0.9, cex.axis = 0.8)

# Plot ROC curves
plot(roc_curve_2, col = "blue", lwd = 2, main = "ROC Curves for Selected Model", 
     xlab = "False Positive Rate", ylab = "True Positive Rate")
lines(roc_curve_val2, col = "red", lwd = 2)
lines(roc_curve_test, col = "green", lwd = 2)
legend("bottomright", legend = c("Training", "Validation", "Test"), col = c("blue", "red", "green"), lwd = 2)

# Add legend with AUC values
legend("bottomright", legend = c(paste("Training (AUC =", round(auc_2, 3), ")"),
                                 paste("Validation (AUC =", round(auc_val2, 3), ")"),
                                 paste("Test (AUC =", round(auc_test, 3), ")")),
       col = c("blue", "red", "green"), lwd = 2)






