# ------------------------------------------------------------------------------
# Visual model evaluation with the ROC curve
# ------------------------------------------------------------------------------


# 1. Draw ROC curve for the kNN with 5 neighbors
# (x-axis: fpr, y-axis: tpr)
# ------------------------------------------------------------------------------

# Load R packages
library(ROCR)

# Load data
d <- readRDS(file = "d_with_predictions.rds")  

pred <- prediction(d$prediction.knn5.prob, d$purchased)
roc.curve <- performance(prediction.obj = pred,
                         measure = "tpr",
                         x.measure = "fpr")
plot(roc.curve)
abline(a=0, b= 1)


# 2. Calculate AUC
# ------------------------------------------------------------------------------
auc.perf = performance(prediction.obj = pred,
                       measure = "auc")
auc.perf@y.values

# Alternative
performance(prediction.obj = pred,
            measure = "auc")@y.values[[1]]

