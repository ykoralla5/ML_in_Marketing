# ------------------------------------------------------------------------------
# SVM with soft margins and linearly separable data
# ------------------------------------------------------------------------------


# 1. Compare the performance of the SVM to the performance of
# a.	kNN
# b.	Na?ve Bayes
# c.  Decision Trees
# d.  Random Forest
# ------------------------------------------------------------------------------

# Load R packages
library(ROCR)
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Set seed
set.seed(123)

# Rename levels of categorical variables to avoid problem when calculating SVM probablities
# https://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret
table(d.training$city)
table(d.test$city)

levels(d.training$city) <- c("City_A", "London", "New_York", "City_B")
levels(d.test$city) <- c("City_A", "London", "New_York", "City_B")

table(d.training$purchased)
table(d.test$purchased)

levels(d.training$purchased) <- c("not_purchased", "purchased")
levels(d.test$purchased) <- c("not_purchased", "purchased")

## SVM - Hyperparameter Optimization
# Model
results.svm <- caret::train(purchased ~ gender + age + city
                     + socioeconomic_status 
                     + sum_of_previous_purchases 
                     + number_of_family_members_who_purchased, 
                     data = d.training, 
                     method = "svmPoly",
                     trControl=trainControl(method = "cv", classProbs =  TRUE))

# Prediction - Probability
d.test$svm <- predict(object=results.svm, newdata=d.test, type="prob")[, 2] 

# to display your prediction in a dataframe fromat (from package "prediction")
pred.svm  <- prediction(d.test$svm, d.test$purchased)

# Code from previous updated to show the performance measures
# of the model with the optimized hyperparameters
# -------------------------------------------------------------------

## Random Forest - Hyperparameter Optimization
# Model
grid <- expand.grid(mtry=2:4) # Default is "ceiling(sqrt(nvar))"

results.rf.opt <- train(purchased ~ gender + age + city
                        + socioeconomic_status 
                        + sum_of_previous_purchases 
                        + number_of_family_members_who_purchased, 
                        data = d.training,
                        method = "cforest",
                        tuneGrid = grid,
                        metric = "Accuracy",
                        trControl=trainControl(method = "cv"))

# Prediction - Probability
d.test$rf.opt <- predict(object=results.rf.opt, newdata=d.test, type = "prob")[, 2]  
pred.rf.opt  <- prediction(d.test$rf.opt, d.test$purchased)

## Random Forest
# Model
results.rf <- train(purchased ~ gender + age + city
                    + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "cforest",
                    trControl=trainControl(method = "cv"))

# Prediction - Probability
d.test$rf <- predict(object=results.rf, newdata=d.test, type = "prob")[, 2]  
pred.rf <- prediction(d.test$rf, d.test$purchased)

## Decision Tree - Optimized
# Model
grid <- expand.grid(cp=seq(0,1,by=0.01))
results.dt.opt <- train(purchased ~ gender + age + city
                        + socioeconomic_status 
                        + sum_of_previous_purchases 
                        + number_of_family_members_who_purchased, 
                        data = d.training, 
                        method = "rpart",
                        tuneGrid = grid,
                        metric = "Accuracy",
                        trControl=trainControl(method = "cv"))

# Prediction - Probability
d.test$dt.opt <- predict(object=results.dt.opt, newdata=d.test, type = "prob")[, 2] 
pred.dt.opt <- prediction(d.test$dt.opt, d.test$purchased)

## Decision Tree
# Model
results.dt <- train(purchased ~ gender + age + city
                    + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "rpart",
                    trControl=trainControl(method = "cv"))

# Prediction - Probability
d.test$dt <- predict(object=results.dt, newdata=d.test, type = "prob")[, 2] 
pred.dt <- prediction(d.test$dt, d.test$purchased)

## KNN
# Model
results.knn <- train(purchased ~ gender + age + city
                     + socioeconomic_status 
                     + sum_of_previous_purchases 
                     + number_of_family_members_who_purchased, 
                     data = d.training, 
                     method = "knn",
                     trControl=trainControl(method = "cv"))

# Prediction - Class Membership
d.test$knn <- predict(object=results.knn, newdata=d.test, type = "prob")[, 2] 
pred.knn <- prediction(d.test$knn, d.test$purchased)

## Naive Bayes
# Model
results.nb <- train(purchased ~ gender + age + city + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "naive_bayes", 
                    tuneGrid=data.frame(laplace=TRUE, usekernel=FALSE, adjust=1), 
                    trControl=trainControl("cv"))

# Prediction - Class Membership
d.test$nb <- predict(object=results.nb, newdata=d.test, type = "prob")[, 2]
pred.nb <- prediction(d.test$nb, d.test$purchased)

# ROC plot
# Decision Tree
roc.dt <- performance(pred.dt,"tpr","fpr")
plot(roc.dt, col="green", print.auc = TRUE)

# Line of classification by chance
abline(a=0, b=1, col="grey")

# Add ROC curves of other models
# KNN
roc.knn <- performance(pred.knn,"tpr","fpr")
plot(roc.knn, add=TRUE, col="red")

# Naive Bayes
roc.nb <- performance(pred.nb,"tpr","fpr")
plot(roc.nb, add=TRUE, col="blue")

# Decision Tree - Hyperparameter Optimization
roc.dt.opt <- performance(pred.dt.opt,"tpr","fpr")
plot(roc.dt.opt, add=TRUE, col="green", lty=2)

# Random Forest
roc.rf <- performance(pred.rf,"tpr","fpr")
plot(roc.rf, add=TRUE, col="purple")

# Random Forest - Hyperparameter Optimization
roc.rf.opt <- performance(pred.rf.opt,"tpr","fpr")
plot(roc.rf.opt, add=TRUE, col="purple", lty=2)

# SVM
roc.svm <- performance(pred.svm,"tpr","fpr")
plot(roc.svm, add=TRUE, col="orange")


# 2. AUC value for each method
# ------------------------------------------------------------------------------
auc.dt <- performance(pred.dt, measure = "auc")
auc.knn <- performance(pred.knn, measure = "auc")
auc.nb <- performance(pred.nb, measure = "auc")
auc.dt.opt <- performance(pred.dt.opt, measure = "auc")
auc.rf <- performance(pred.rf, measure = "auc")
auc.rf.opt <- performance(pred.rf.opt, measure = "auc")
auc.svm <- performance(pred.svm, measure = "auc")

data.frame(Model=c("Decision Tree", 
                   "KNN", 
                   "Naive Bayes", 
                   "Decision Tree (opt.)", 
                   "Random Forest", 
                   "Random Forest (opt.)",
                   "SVM"),
           AUC=c(auc.dt@y.values[[1]], 
                 auc.knn@y.values[[1]], 
                 auc.nb@y.values[[1]], 
                 auc.dt.opt@y.values[[1]], 
                 auc.rf@y.values[[1]],
                 auc.rf.opt@y.values[[1]],
                 auc.svm@y.values[[1]]))
