# ------------------------------------------------------------------------------
# Visual model evaluation with the Gain and Lift Charts
# ------------------------------------------------------------------------------


# 1. Gain chart
# ------------------------------------------------------------------------------

# Load R packages
library(ROCR)

# Load data
d <- readRDS(file = "d_with_predictions.rds")  

# computing predictions
pred <- prediction(d$prediction.knn5.prob, d$purchased)

gain.chart <- performance(prediction.obj = pred,
                          measure = "tpr",
                          x.measure = "rpp")
plot(gain.chart)


# 2. Lift chart
# ------------------------------------------------------------------------------
lift.chart <- performance(prediction.obj = pred,
                          measure = "lift",
                          x.measure = "rpp")
plot(lift.chart)
