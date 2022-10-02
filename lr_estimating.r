# ------------------------------------------------------------------------------
# Estimating Logistic Regression
# ------------------------------------------------------------------------------


# 1. Figure out which function to use for estimating logistic regression
# ------------------------------------------------------------------------------

# Have a look at the following website:
# http://topepo.github.io/caret/train-models-by-tag.html

# For this exercise we decide to use the train() command 
# in combination with the argument: method="LogitBoost" 


# 2. Apply a logistic regression to the training data
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Set seed
# RNGkind(sample.kind = "Rounding") # Ensure consistent sampling across R versions
set.seed(123, kind = "Mersenne-Twister", normal.kind = "Inversion")

# Train the model
results.lr <- train(purchased ~ gender + age + city
                    + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "LogitBoost",
                    trControl=trainControl(method = "cv"))


# 3. Prediction - Class Membership
# ------------------------------------------------------------------------------
d.test$lr <- predict(object=results.lr, newdata=d.test) 


# 4. Performance comparison
# ------------------------------------------------------------------------------
tab.lr <- table(d.test$lr, d.test$purchased, dnn = c("pred", "actual"))
confusionMatrix(tab.lr, positive = "1")

# In-sample-performance
# confusionMatrix(table(predict(results.dt), d.training$purchased,
# dnn = c("pred", "actual")), positive = "1")
