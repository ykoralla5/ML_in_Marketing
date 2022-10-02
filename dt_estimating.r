# ------------------------------------------------------------------------------
# Estimating Decision Trees
# ------------------------------------------------------------------------------


# 1. Figure out which functions to use for estimating decision trees.
# ------------------------------------------------------------------------------

# Have a look at the following website:
# http://topepo.github.io/caret/train-models-by-tag.html

# For this exercise we decide to use the train() command 
# in combination with the argument: method="rpart" 
#  CART stands for "classification and regression trees"


# 2. Apply a decision tree to the training data
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Set seed
RNGkind(sample.kind = "Rounding") # Ensure consistent sampling across R versions
set.seed(123)

results.dt <- train(purchased ~ gender + age + city
                    + socioeconomic_status 
                    + sum_of_previous_purchases 
                    + number_of_family_members_who_purchased, 
                    data = d.training, 
                    method = "rpart",
                    trControl=trainControl(method = "cv"))


# 3. Prediction - Class Membership
# ------------------------------------------------------------------------------

d.test$dt <- predict(object=results.dt, newdata=d.test) 


# 4. Performance comparison
# ------------------------------------------------------------------------------
tab.dt <- table(d.test$dt, d.test$purchased, dnn = c("pred", "actual"))
confusionMatrix(tab.dt, positive = "1")

predict(object=results.dt, 
        newdata=data.frame(gender="male",
                           age=30,
                           city="New York",
                           socioeconomic_status="low",
                           sum_of_previous_purchases=100,
                           number_of_family_members_who_purchased=2))

# In-sample-performance
# confusionMatrix(table(predict(results.dt),
# d.training$purchased, dnn = c("pred", "actual")), positive = "1")

