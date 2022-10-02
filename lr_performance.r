# ------------------------------------------------------------------------------
# Logistic Regression: Performance measures
# ------------------------------------------------------------------------------


# 1. Use ROC curves to compare the performance of the logistic regression
# to the performance of
# a.	kNN
# b.	Na?ve Bayes
# c.  Decision trees
# d.  Random forest
# e.  SVM
# ------------------------------------------------------------------------------

# Load R packages
# install.packages("caretEnsemble")
library(ROCR)
library(caret)
library(caretEnsemble)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# replace blanks by "_" (as this might cause problems in later models)
levels(d.training$city) <- c("City_A", "London", "New_York", "City_B")
levels(d.test$city) <- c("City_A", "London", "New_York", "City_B")

levels(d.training$purchased) <- c("not_purchased", "purchased")
levels(d.test$purchased) <- c("not_purchased", "purchased")

# create submodels
# caretList is the preferred way to construct list of caret models in this package, 
# as it will ensure the resampling indexes are identical across all models. 

# Set seed
# RNGkind(sample.kind = "Rounding") # Ensure consistent sampling across R versions
set.seed(123,kind = "Mersenne-Twister", normal.kind = "Inversion")

models <- caretList(purchased ~ gender + age + city 
                    + socioeconomic_status + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    metric="ROC",
                    trControl = trainControl(method = "boot",
                                             index = createFolds(d.training$purchased),
                                             savePredictions = "all", classProbs=T,
                                             summaryFunction = twoClassSummary), # Using "defaultSummary" instead, will output Kappa and Accuracy.
                    tuneList = list(
                      lr = caretModelSpec(method = "LogitBoost"),
                      svm.opt = caretModelSpec(method = "svmPoly",
                                               tuneGrid = expand.grid(degree=1:2,
                                                                      scale=seq(0.8,1.2,by=0.1), C=1)),
                      svm = caretModelSpec(method = "svmPoly"),
                      rf.opt = caretModelSpec(method = "rf",
                                              tuneGrid = expand.grid(mtry=2:4)),
                      rf = caretModelSpec(method = "rf"),
                      dt.opt = caretModelSpec(method = "rpart",
                                              tuneGrid = expand.grid(cp=seq(0,1,by=0.1))),
                      dt = caretModelSpec(method = "rpart"),         
                      knn = caretModelSpec(method = "knn"),
                      nb = caretModelSpec(method = "naive_bayes",
                                          tuneGrid=data.frame(laplace=TRUE, usekernel=FALSE, adjust=1))  # Models with detailed settings
                    ))


# Get model results, e.g. Sensitivity, Specificity
# To get Accuracy, set metric="Accuracy" and change  summaryFunction as mention above
model_list_results <- resamples(models) # Get results
summary(model_list_results) # Average accuracy for each model across folds 
dotplot(model_list_results) # Plot average accuracy for each model across folds

# Calculate out-of-sample performance (test set) [ALTERNATIVE]
# install.packages("caTools")
library(caTools)
model_list_preds <- lapply(models, predict, newdata=d.test, type="prob")
model_list_preds <- lapply(model_list_preds, function(x) x[,"purchased"]) # Specify here the name of the relevant level of the dependent variable 
model_list_preds <- data.frame(model_list_preds)

# ROC plot [ALTERNATIVE]
library(ROCR)

# List of model predictions and actual values
preds_list <- as.list(model_list_preds)
actuals_list <- rep(list((as.numeric(d.test$purchased)-1)), length(preds_list))

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:length(preds_list)), main = "Test Set ROC Curves")

legend(x = "bottomright", 
       legend = names(model_list_preds),
       fill = 1:length(preds_list))


# 2. Calculating AUC [ALTERNATIVE]
# ------------------------------------------------------------------------------
caTools::colAUC(model_list_preds, d.test$purchased)
