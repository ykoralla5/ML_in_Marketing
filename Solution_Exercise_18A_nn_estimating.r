# ------------------------------------------------------------------------------
# Estimating Neural Networks
# ------------------------------------------------------------------------------


# 1. Figure out which functions to use for estimating decision trees.
# ------------------------------------------------------------------------------

# Have a look at the following website:
# http://topepo.github.io/caret/train-models-by-tag.html

# For this exercise we decide to use the train() command 
# in combination with the argument: method="nnet"


# 2. Apply a neural network to the training data
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Set seed
# RNGkind(sample.kind = "Rounding") # Ensure consistent sampling across R versions
set.seed(123)

# Rename levels of categorical variables to avoid problem when calculating SVM probabilities
# https://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret
table(d.training$city)
table(d.test$city)

levels(d.training$city) <- c("City_A", "London", "New_York", "City_B")
levels(d.test$city) <- c("City_A", "London", "New_York", "City_B")

table(d.training$purchased)
table(d.test$purchased)

levels(d.training$purchased) <- c("not_purchased", "purchased")
levels(d.test$purchased) <- c("not_purchased", "purchased")


# Train the model
results.nn <- train(purchased ~ gender + age + city
                    + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "nnet",
                    trControl=trainControl(method = "cv"))


# 3. Prediction - Class Membership
# ------------------------------------------------------------------------------
d.test$nn <- predict(object=results.nn, newdata=d.test) 


# 4. Performance comparison
# ------------------------------------------------------------------------------
tab.nn <- table(d.test$nn, d.test$purchased, dnn = c("pred", "actual"))
confusionMatrix(tab.nn, positive = "purchased")

# In-sample-performance
# confusionMatrix(table(predict(results.dt), d.training$purchased, dnn = c("pred", "actual")), positive = "1")
