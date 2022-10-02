# ------------------------------------------------------------------------------
# Apply hyperparameter optimization to KNN
# ------------------------------------------------------------------------------


# 1. What are the hyperparameters of the kNN method?
# ------------------------------------------------------------------------------

# Have a look at the following website:
# http://topepo.github.io/caret/train-models-by-tag.html.
# For the implementation method = "knn", there is only one hyperparameter:
# k (#Neighbors).


# 2. Optimization of hyperparameters through grid search
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Train the model
set.seed(123)
results.knn.grid <- train(purchased ~ gender + age + city + socioeconomic_status 
                          + sum_of_previous_purchases 
                          + number_of_family_members_who_purchased, 
                          data = d.training, 
                          method = "knn", 
                          tuneGrid = data.frame(k=1:30), 
                          trControl = trainControl(method = "LGOCV",
                                                 p = 0.8,
                                                 number = 50,
                                                 savePredictions = T))

# Summarize the model
results.knn.grid

# Visualization of Performance in training sample
plot(results.knn.grid)

# Prediction 
# (a) Class Membership
d.test$prediction.knn.gridclass <- predict(object = results.knn.grid,
                                           newdata = d.test)

# (b) Probability
d.test$prediction.knn.gridprob <- predict(object = results.knn.grid, 
                                         newdata = d.test, 
                                         type = "prob")[, 2]

# Performance metrics
tab.knn.grid <- table(d.test$prediction.knn.gridclass, d.test$purchased)
confusionMatrix(tab.knn.grid, positive = "1")



