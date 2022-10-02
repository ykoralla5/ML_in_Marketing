# ------------------------------------------------------------------------------
# Using sampling to cope with class imbalance
# ------------------------------------------------------------------------------


# (a) Oversample the minority class
# ------------------------------------------------------------------------------

# Load data
load(file = "d_splitted_in_train_and_test.rdata")

set.seed(123)
results.knn.5 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       tuneGrid=data.frame(k=5), 
                       trControl=trainControl(method = "LGOCV", p = 0.1,
                                              sampling="up",
                                              savePredictions = T))

results.knn.5


# (b) Undersample the majority class
# ------------------------------------------------------------------------------
set.seed(123)
results.knn.6 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       tuneGrid=data.frame(k=5), 
                       trControl=trainControl(method = "LGOCV", p = 0.1,
                                              sampling="down",
                                              savePredictions = T))

results.knn.6


# (c) Synthesize new minority classes
# ------------------------------------------------------------------------------
set.seed(123)
results.knn.7 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       tuneGrid=data.frame(k=5), 
                       trControl=trainControl(method = "LGOCV", p = 0.1,
                                              sampling="smote",
                                              savePredictions = T))

results.knn.7
