# ------------------------------------------------------------------------------
# SVM with hard margins
# ------------------------------------------------------------------------------


# 1. Figure out which functions to use for estimating a support vector machine
# ------------------------------------------------------------------------------

# Have a look at the following website:
# http://topepo.github.io/caret/train-models-by-tag.html

# For this exercise we decide to use the train() command 
# in combination with the argument: method="svmPoly" 


# 2. Apply a SVM to the training data
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

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

# Set seed
set.seed(123)

# Train the model
results.svm <- train(purchased ~ gender + age + city
                    + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "svmPoly",
                    trControl=trainControl(method = "cv"))


# 3. Prediction - Class Membership
# ------------------------------------------------------------------------------
d.test$svm <- predict(object=results.svm, newdata=d.test) 


# 4. Performance comparison
# ------------------------------------------------------------------------------
tab.svm <- table(d.test$svm, d.test$purchased, dnn = c("pred", "actual"))
caret::confusionMatrix(tab.svm, positive = "purchased")

# In-sample-performance
# confusionMatrix(table(predict(results.dt), d.training$purchased, dnn = c("pred", "actual")), positive = "1")

