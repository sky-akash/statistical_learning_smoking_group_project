library(readr)
Smoking_Training_Data <- read_csv("Health Dataset/Kaggle_Smoking/Smoking_Training_Data.csv")

#Loading Libraries
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(pscl)
library(car)


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

###################################################################
# Removing Outliers Or Not ? (Let's not remove it)
# Uncomment the lines below to Remove Outliers and Perform PCA

# df_fe <- df_fe_unclean[!(df_fe_unclean$Cholesterol > 240 |
#                            df_fe_unclean$LDL > 190 |
#                            df_fe_unclean$eyesight_left > 2 |
#                            df_fe_unclean$eyesight_right > 2 |
#                            df_fe_unclean$fasting_blood_sugar > 126 |
#                            df_fe_unclean$systolic > 160 |
#                            df_fe_unclean$relaxation > 100 |
#                            df_fe_unclean$triglyceride > 200 |
#                            df_fe_unclean$HDL < 30 | df_fe_unclean$HDL > 120 |
#                            df_fe_unclean$hemoglobin > 19 |
#                            df_fe_unclean$serum_creatinine > 1.6 |
#                            df_fe_unclean$AST < 9 | df_fe_unclean$AST > 60 |
#                            df_fe_unclean$ALT < 6 | df_fe_unclean$ALT > 60 |
#                            df_fe_unclean$Gtp > 180), ]
#################################################################
# Also, comment  the line below to work on data after removal of Outliers
df_fe <- df_fe_unclean


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

#Code between the following line Mark
#................................................................................#
#In this section, PCA on Whole data including Feature Engineered Varaibles
#
##########################################
# numeric_cols_df_fe <- sapply(df_fe, is.numeric)
# numeric_df_fe <- df_fe[, numeric_cols_df_fe]
# 
# col_means <- apply(numeric_df_fe, 2, mean)
# col_means
# 
# col_var <- round(apply(numeric_df_fe, 2, var),3)
# col_var
# 
# pr.out = prcomp(numeric_df_fe, scale = TRUE)
# names(pr.out)
# 
# pr.out$rotation
# 
# dim(pr.out$x)
# 
# par(mfrow = c(1,1))
# biplot(pr.out, scale=0)
# 
# 
# #....................
# 
# pr.out$rotation = -pr.out$rotation
# pr.out$x = -pr.out$x
# biplot(pr.out, scale=0)
# 
# pr.out$sdev
# 
# pr.var = pr.out$sdev^2
# pr.var
# 
# pve = round(pr.var/sum(pr.var),4)
# pve
# 
# plot(pve, xlab="Principal Component", ylab="Proportion of variance explained", ylim=c(0,1), type='b')
# 
# plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1), type='b')

#................................................................................#


###################################################################################################

#Code between the following line Mark (No Outliers removed)
###................................................................................####
#In this section, PCA on Selected Variables and removing redundant Variables as our knowledge from Logistic Regression. #


dim(df_fe)
str(df_fe)

# Keep Only numeric Variable sin df_fe
df_fe.labs = df_fe$smoking
numeric_cols_df_fe <- sapply(df_fe, is.numeric)
df_fe.data <- df_fe[, numeric_cols_df_fe]

cols_to_exclude <- c("height_cm", "weight_kg", "waist_cm", "eyesight_left", 
                     "eyesight_right", "systolic", "relaxation", "LDL", "AST", 
                     "ALT", "BMI")

# Selecting only Specific Columns for Proceeding with the PCA
df_fe.data <- df_fe.data[, !names(df_fe.data) %in% cols_to_exclude]

str(df_fe.data)

table(df_fe.labs)


#Apply PCA
pr.out = prcomp(df_fe.data, scale=TRUE)

Cols=function(vec){
  cols=rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])
}

par(mfrow=c(1,2))

plot(pr.out$x[,1:2], col=Cols(df_fe.labs), pch=19, xlab="Z1", ylab="Z2")
plot(pr.out$x[,1:3], col=Cols(df_fe.labs), pch=19, xlab="Z1", ylab="Z3")
plot(pr.out$x[,1:6], col=Cols(df_fe.labs), pch=19, xlab="Z1", ylab="Z6")
plot(pr.out$x[,1:10], col=Cols(df_fe.labs), pch=19, xlab="Z1", ylab="Z10")


summary(pr.out)

par(mfrow=c(1,1))
plot(pr.out)


par(mfrow=c(1,2))
#Variance explained by Different PCs
pve = summary(pr.out)$importance[2,]
#Plot Scree Plots
plot(pve, type='o', ylab="PVE", xlab="Principal Components", col="tan")


#Cumulative Variance
cumsum_pve = summary(pr.out)$importance[3,]
#Plot Scree Plots
plot(cumsum_pve, type='o', ylab="PVE", xlab="Principal Components", col="maroon2")


#Clustering the data using K-Means on the first 8 Principal Components

# Extract the first 8 principal components
pc_sel <- pr.out$x[, 1:8]

# Perform K-Means clustering on the first 8 principal components
set.seed(123)  # For reproducibility
kmeans_out <- kmeans(pc_sel, centers = 2)  # Assuming 2 clusters, adjust centers as needed

# Get cluster assignments
kmeans_clusters <- kmeans_out$cluster

par(mfrow=c(1,1))
# Visualize the clusters obtained from K-Means
plot(pc_sel[, c(1, 2)], col = c("#FF5000", "#33FF57")[kmeans_clusters], pch = 19,
     main = "K-Means Clustering of First 8 Principal Components",
     xlab = "PC1", ylab = "PC2")

# Add real smoking labels
points(pc_sel[, c(1, 2)], col = c("#0072B2", "#E69F00")[df_fe.labs], pch = 19)

legend("topright", legend = c("Smoker", "Non-Smoker"), col = c("#0072B2", "#E69F00"), pch = 19, bty = "n")

# Compare K-Means clusters with real smoking labels
table(kmeans_clusters, df_fe.labs)

#Silhoutte score
library(cluster)

pc_dist <- dist(pc_sel)

# Silhouette Score
silhouette_scores <- silhouette(kmeans_clusters, pc_dist)
silhouette_widths <- summary(silhouette_scores)$clus.avg.width
silhouette_score <- mean(silhouette_widths)
cat("Silhouette Score:", silhouette_score, "\n")















################################
#practive lines under this 
# silhouette_score <- silhouette(kmeans_clusters, pc_dist)$avg.width
# 
# # Adjusted Rand Index
# adjusted_rand_index <- adjustedRand(kmeans_clusters, as.numeric(df_fe.labs))
# 
# # Output the metrics
# print("Silhouette Score:", silhouette_score, "\n")
# print("Adjusted Rand Index:", adjusted_rand_index, "\n")
# 
# 
# 
# ##########
# smokers <- df_fe[df_fe$smoking == "1", ]
# non_smokers <- df_fe[df_fe$smoking == "0", ]
# 
# numeric_cols_df_fe <- sapply(df_fe, is.numeric)
# numeric_df_fe <- df_fe[, numeric_cols_df_fe]
# 
# str(numeric_df_fe)
# 
# cols_to_exclude <- c("age", "height_cm", "weight_kg", "waist_cm", "systolic", "relaxation", "LDL", "AST", "ALT", "BMI")
# 
# numeric_df_fe <- numeric_df_fe[, !(names(numeric_df_fe) %in% cols_to_exclude)]
# 
# str(numeric_df_fe)
# 
# col_means <- apply(numeric_df_fe, 2, mean)
# col_means
# 
# col_var <- round(apply(numeric_df_fe, 2, var),3)
# col_var
# 
# pr.out = prcomp(numeric_df_fe, scale = TRUE)
# names(pr.out)
# 
# pr.out$rotation
# 
# dim(pr.out$x)
# 
# # Extract the scores of the first 8 principal components
# pc_scores <- pr.out$x[, 1:8]
# 
# # Extract the loadings of the first 8 principal components
# pc_loadings <- pr.out$rotation[, 1:8]
# 
# # Plot the biplot for selected 8 PC and Overlaying smokers and non-smokers with different colors
# biplot(pc_scores, pc_loadings, scale=0)
# points(pc_scores[df_fe$smoking == "1", ], col = "lightblue", pch = 19) # Smokers in blue
# points(pc_scores[df_fe$smoking == "0", ], col = "pink", pch = 19) # Non-smokers in pink
# 
# 
# par(mfrow = c(1,1))
# biplot(pr.out, scale=0)
# 
# 
# 
# 
# #Below Code is to Plot the smokers and NOn Smokers on the Biplot
# 
# #...............................................................................
# if(nrow(pc_scores) != nrow(df_fe)) {
#   stop("Number of rows in pc_scores and df_fe do not match.")
# }
# 
# 
# if("1" %in% df_fe$smoking) {
#   points(pc_scores[df_fe$smoking == "1", ], col = "blue", pch = 19) # Smokers in blue
# }
# if("0" %in% df_fe$smoking) {
#   points(pc_scores[df_fe$smoking == "0", ], col = "pink", pch = 19) # Non-smokers in pink
# }
# #...............................................................................
# #...
# 
# pr.out$rotation = -pr.out$rotation
# pr.out$x = -pr.out$x
# biplot(pr.out, scale=0)
# 
# pr.out$sdev
# 
# pr.var = pr.out$sdev^2
# pr.var
# 
# pve = round(pr.var/sum(pr.var),4)
# pve
# 
# plot(pve, xlab="Principal Component", ylab="Proportion of variance explained", ylim=c(0,1), type='b')
# 
# plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1), type='b')
# 

























