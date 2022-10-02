# ------------------------------------------------------------------------------
# Decision Trees: Performance measures
# ------------------------------------------------------------------------------


# 1. Use ROC curves to compare the performance of the decision trees
# to the performance of
# a.	kNN
# b.	Naïve Bayes.
# ------------------------------------------------------------------------------

# Load R packages
library(ROCR)
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Set seed
RNGkind(sample.kind = "Rounding") # Ensure consistent sampling across R versions
set.seed(123)

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

## kNN
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
                    tuneGrid=data.frame(laplace=TRUE, usekernel=FALSE, adjust=1), 
                    trControl=trainControl("cv"))

# Prediction - Probability
d.test$nb <- predict(object=results.nb, newdata=d.test, type = "prob")[, 2]
pred.nb <- prediction(d.test$nb, d.test$purchased)

# ROC plot
# Setup ROC plot
roc.dt <- performance(pred.dt,"tpr","fpr")
plot(roc.dt, color="green", print.auc = TRUE)

# Line of classification by chance
abline(a=0, b=1, col="grey")

# Add ROC curves of other models
# kNN
roc.knn <- performance(pred.knn,"tpr","fpr")
plot(roc.knn, add=TRUE, col="red")

# Naive Bayes
roc.nb <- performance(pred.nb,"tpr","fpr")
plot(roc.nb, add=TRUE, col="blue")


# 2. AUC value for each method
# ------------------------------------------------------------------------------

auc.dt <- performance(pred.dt, measure = "auc")
auc.knn <- performance(pred.knn, measure = "auc")
auc.nb <- performance(pred.nb, measure = "auc")

data.frame(Model=c("Decision Tree", "KNN", "Naive Bayes"),
           AUC=c(auc.dt@y.values[[1]],
                 auc.knn@y.values[[1]],
                 auc.nb@y.values[[1]]))
