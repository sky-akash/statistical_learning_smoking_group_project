attach(sl_smoking_kg)

colnames(sl_smoking_kg)

variables_of_interest <- c("age", "systolic", "relaxation", "fasting blood sugar", 
                           "Cholesterol", "Urine protein", "serum creatinine", "AST", "smoking")

subset_data <- sl_smoking_kg[, variables_of_interest]

subset_data <- na.omit(subset_data)

pairs(subset_data)