# ------------------------------------------------------------------------------
# Random Forest: Estimating random forest
# ------------------------------------------------------------------------------


# 1. Figure out which functions to use for estimating random forest
# ------------------------------------------------------------------------------

# Have a look at the following website:
# http://topepo.github.io/caret/train-models-by-tag.html

# For this exercise we decide to use the train() command 
# in combination with the argument: method="rf" 


# 2. Apply a random forest model to the training data
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Set seed
RNGkind(sample.kind = "Rounding") # Ensure consistent sampling across R versions
set.seed(123)

# Train the model
results.rf <- train(purchased ~ gender + age + city
                    + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "rf",
                    trControl=trainControl(method = "cv"))


# 3. Prediction - Class Membership
# ------------------------------------------------------------------------------
d.test$rf <- predict(object=results.rf, newdata=d.test) 


# 4. Performance comparison
# ------------------------------------------------------------------------------
tab.rf <- table(d.test$rf, d.test$purchased, dnn = c("pred", "actual"))
confusionMatrix(tab.rf, positive = "1")

# In-sample-performance
# confusionMatrix(table(predict(results.rf), d.training$purchased, dnn = c("pred", "actual")), positive = "1")

