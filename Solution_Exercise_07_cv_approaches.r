# ------------------------------------------------------------------------------
# Cross-Validation Approaches
# ------------------------------------------------------------------------------


# 1. Apply cross-validation methods to the kNN example
# ------------------------------------------------------------------------------

# Load packages
library("caret")

# Load data
load(file = "d_splitted_in_train_and_test.rdata")


# (a) Exhaustive cross validation
# ------------------------------------------------------------------------------

# Leave-p-out cross validation
set.seed(123)

results.knn.3 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       tuneGrid = data.frame(k=5), 
                       trControl = trainControl(method = "LGOCV", p = 0.9,
                                              savePredictions = T))

# Results
results.knn.3

# Some detailed results on the sampling
results.knn.3$control$index # Training samples
results.knn.3$control$indexOut # Holdout samples
results.knn.3$resample # Performance measures for each resample set
mean(results.knn.3$resample$Accuracy) # Replicate results from model fit output
results.knn.3$resampledCM # Confusion matrix for each resample set
resampleHist(results.knn.3) # Histogram for confusion matrix results for each resample set



# (b) Non-exhaustive cross validation
# ------------------------------------------------------------------------------

# K-Fold cross validation
set.seed(123)
results.knn.4 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       tuneGrid = data.frame(k=5), 
                       trControl = trainControl(method = "cv",
                                              savePredictions = T))

# Results
results.knn.4

# Some detailed results on the sampling
results.knn.4$control$index # Training samples
results.knn.4$control$indexOut # Holdout samples
results.knn.4$resample # Performance measures for each resample set
mean(results.knn.4$resample$Accuracy) # Replicate results from model fit output
results.knn.4$resampledCM # Confusion matrix for each resample set
resampleHist(results.knn.4) # Histogram for confusion matrix results for each resample set


