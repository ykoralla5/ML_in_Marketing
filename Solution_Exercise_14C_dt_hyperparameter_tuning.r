# ------------------------------------------------------------------------------
# Decision Trees: Hyperparameter tuning
# ------------------------------------------------------------------------------


#  1. Tuning hyperparameters of the decision tree
# ------------------------------------------------------------------------------

# Load R packages
library(ROCR)
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Set seed
RNGkind(sample.kind = "Rounding") # Ensure consistent sampling across R versions
set.seed(123)

# Decision Tree
results.dt.opt <- train(purchased ~ gender + age + city
                        + socioeconomic_status 
                        + sum_of_previous_purchases 
                        + number_of_family_members_who_purchased, 
                        data = d.training, 
                        method = "rpart",
                        tuneGrid = data.frame(cp=seq(0,1,by=0.1)), #shows grid search
                        metric = "Accuracy",
                        trControl=trainControl(method = "cv"))

# Prediction - Probability
d.test$dt.opt <- predict(object=results.dt.opt, newdata=d.test, type = "prob")[, 2] 
pred.dt.opt <- prediction(d.test$dt.opt, d.test$purchased)


# Code from previous updated to show the performance measures
# of the model with the optimized hyperparameters
# -------------------------------------------------------------------

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
d.test$dt <- predict(object=results.dt,
                     newdata=d.test, type = "prob")[, 2] 
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

# Prediction - Probability
d.test$knn <- predict(object=results.knn, newdata=d.test, type = "prob")[, 2] 
pred.knn <- prediction(d.test$knn, d.test$purchased)

## Naive Bayes
# Model
results.nb <- train(purchased ~ gender + age + city + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "naive_bayes", 
                    tuneGrid=data.frame(laplace=0, usekernel=FALSE, adjust=1), 
                    trControl=trainControl("cv"))

# Prediction - Probability
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


# 2. Add ROC curve of tuned model to chart
# -------------------------------------------------------------------

# Decision Tree - Hyperparameter Optimization
roc.dt.opt <- performance(pred.dt.opt,"tpr","fpr")
plot(roc.dt.opt, add=TRUE, col="green", lty=2)

# AUC values
auc.dt <- performance(pred.dt, measure = "auc")
auc.knn <- performance(pred.knn, measure = "auc")
auc.nb <- performance(pred.nb, measure = "auc")
auc.dt.opt <- performance(pred.dt.opt, measure = "auc")

data.frame(Model=c("Decision Tree",
                   "KNN",
                   "Naive Bayes",
                   "Decision Tree (opt.)"),
           AUC=c(auc.dt@y.values[[1]],
                 auc.knn@y.values[[1]],
                 auc.nb@y.values[[1]],
                 auc.dt.opt@y.values[[1]]))


