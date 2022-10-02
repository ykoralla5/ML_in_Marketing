# ------------------------------------------------------------------------------
# Voting and Stacking
# ------------------------------------------------------------------------------

# 1. Train the models
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")

set.seed(123)
results.knn.5.ensemble <- train(purchased ~ gender + age + city
                          + socioeconomic_status 
                          + sum_of_previous_purchases 
                          + number_of_family_members_who_purchased, 
                          data = d.training, 
                          method = "knn", 
                          tuneGrid = data.frame(k=5), 
                          trControl = trainControl(method = "none"))

# Prediction - Class Membership
d.training$knn.5 <- predict(object = results.knn.5.ensemble) 

# Prediction - Class Membership
d.test$knn.5 <- predict(object = results.knn.5.ensemble, newdata = d.test)

# KNN (k=19)
results.knn.19.ensemble <- train(purchased ~ gender + age + city
                              + socioeconomic_status 
                              + sum_of_previous_purchases 
                              + number_of_family_members_who_purchased, 
                              data = d.training, 
                              method = "knn", 
                              tuneGrid = data.frame(k=19), 
                              trControl = trainControl(method = "none"))

# Prediction - Class Membership
d.training$knn.19 <- predict(object = results.knn.19.ensemble)

# Prediction - Class Membership
d.test$knn.19 <- predict(object = results.knn.19.ensemble,
                         newdata = d.test) 

# Naive Bayes
results.nb.ensemble <- train(purchased ~ gender + age + city
                      + socioeconomic_status 
                      + sum_of_previous_purchases 
                      + number_of_family_members_who_purchased, 
                      data = d.training, 
                      method = "naive_bayes", 
                      tuneGrid = data.frame(laplace=T, usekernel=FALSE, adjust=1), 
                      trControl = trainControl("none"))

# Prediction - Class Membership
d.training$nb <- predict(object = results.nb.ensemble)

# Prediction - Class Membership
d.test$nb <- predict(object = results.nb.ensemble, newdata = d.test) 

# Ensemble classifier using kNN
results.ensemble.combiner <- train(purchased ~ knn.5 + knn.19 + nb, 
                             data = d.training, 
                             method = "knn", 
                             tuneGrid = data.frame(k=51),
                             trControl = trainControl("none"))

# Prediction - Class Membership
d.training$ensemble <- predict(object = results.ensemble.combiner)

# Prediction - Class Membership
d.test$ensemble <- predict(object = results.ensemble.combiner,
                           newdata = d.test) 

# Overview of all predictions
head(d.test,20)

# Performance comparison
# kNN (k = 5)
tab.knn.5 <- table(d.test$knn.5, d.test$purchased,
                   dnn = c("pred", "actual"))
confusionMatrix(tab.knn.5, positive = "1")

# kNN (k = 19)
tab.knn.19 <- table(d.test$knn.19, d.test$purchased,
                    dnn = c("pred", "actual"))
confusionMatrix(tab.knn.19, positive = "1")

# Naive Bayes
tab.nb <- table(d.test$nb, d.test$purchased,
                dnn = c("pred", "actual"))
confusionMatrix(tab.nb, positive = "1")

# Ensemble
tab.ensemble <- table(d.test$ensemble, d.test$purchased,
                      dnn = c("pred", "actual"))
confusionMatrix(tab.ensemble, positive = "1")


