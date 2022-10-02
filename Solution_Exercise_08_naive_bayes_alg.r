# ------------------------------------------------------------------------------
# Naive Bayes estimation
# ------------------------------------------------------------------------------


# 1. Use the function train() to apply the  algorithm
# ------------------------------------------------------------------------------

# Load R packages
library(caret)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")  

# Run the algorithm 
# The "naive_bayes" method in the caret package uses the naivebayes packages, 
# which automatically applies the correct procedures for categorical and continuous 
# variables
results.nb.1 <- train(purchased ~ gender + age + city + socioeconomic_status 
                      + sum_of_previous_purchases 
                      + number_of_family_members_who_purchased, 
                      data = d.training, 
                      method = "naive_bayes", 
                      tuneGrid=data.frame(laplace=0, usekernel=FALSE, adjust=1), 
                      trControl=trainControl("none"))
                      # Alternative as by default naive_bayes tries two models, i.e. "usekernel"= TRUE / FALSE.
                      # This does not work for trainControl("none"), thus we have to fix the parameters in tuneGrid  
                      #tuneGrid=data.frame(laplace=0, usekernel=FALSE, adjust=1), 
                      #trControl=trainControl("cv"))
                      # More information on the Hyperparameter Laplace can be found at:
                      # https://uc-r.github.io/naive_bayes (Section "Laplace Smoother")

results.nb.1 <- train(purchased ~ gender + age + city + socioeconomic_status 
                      + sum_of_previous_purchases 
                      + number_of_family_members_who_purchased, 
                      data = d.training, 
                      method = "naive_bayes", 
                      tuneGrid=data.frame(laplace=0, usekernel=FALSE, adjust=1), 
                      trControl=trainControl("cv"))

# Check what the result object returns when you call it
results.nb.1


# 2. Test the out-of-sample performance 
# ------------------------------------------------------------------------------

# Class Membership
d.test$prediction.nb.1.class <- predict(object=results.nb.1, 
                           newdata=d.test)

# Probability
d.test$prediction.nb.1.prob.1 <- predict(object=results.nb.1, 
                           newdata=d.test, 
                           type = "prob")[, 2]

# Performance metrics
tab.1 <- table(d.test$prediction.nb.1.class, d.test$purchased)
confusionMatrix(tab.1, positive = "1")

