# ------------------------------------------------------------------------------
# Algorithm 1 - k-nearest neighbor 
# ------------------------------------------------------------------------------


# 1. Figure out which functions to use for applying the kNN algorithm.
# ------------------------------------------------------------------------------

# Have a look at the following website:
# http://topepo.github.io/caret/train-models-by-tag.html

# For this exercise we decide to use the train() command 
# in combination with the argument: method="knn" 
# Another possibility would be to use a different implementation,
# e.g.: method = "kknn"


# 2. Apply the kNN algorithm 
# ------------------------------------------------------------------------------
# Load R packages
library(caret)

# Load data
d <- readRDS(file = "d.rds")  

# Train the model (kNN)
results.knn <- train(purchased ~ gender + age + city + socioeconomic_status 
                     + sum_of_previous_purchases 
                     + number_of_family_members_who_purchased, 
                     data = d, # because no training and test data exists for knn
                     method = "knn", 
                     preProcess = "scale",
                     tuneGrid=data.frame(k=5), 
                     trControl=trainControl("none"))

# # Train the model (kNN) - Alternative
# results.knn.temp <- train(purchased ~ ., 
#                        data = d, 
#                        method = "knn", 
#                        tuneGrid=data.frame(k=5), 
#                        trControl=trainControl("none"))

# Check the resulting object 
results.knn


# 3. Prediction for new observation
# ------------------------------------------------------------------------------
predict(object=results.knn, 
        newdata=data.frame(gender="male",
                           age=30,
                           city="New York",
                           socioeconomic_status="low",
                           sum_of_previous_purchases=100,
                           number_of_family_members_who_purchased=2))


# 4. Prediction for entire dataset
# ------------------------------------------------------------------------------
# (a) Class Membership
d$prediction.knn5.class <- predict(object=results.knn)

# (b) Probability
d$prediction.knn5.prob <- predict(object=results.knn, type = "prob")[, 2]

# Save the predictions
saveRDS(object = d, file= "d_with_predictions.rds") # Save as RDS object
# fwrite(d, "d_with_predictions.csv") # Alternatively, save as CSV file
