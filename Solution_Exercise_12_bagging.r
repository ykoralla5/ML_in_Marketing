# ------------------------------------------------------------------------------
# Bagging
# ------------------------------------------------------------------------------


# 1. Apply Bagging
# ------------------------------------------------------------------------------

# Load packages
install.packages("adabag")
library(adabag)

# Load data
load(file = "d_splitted_in_train_and_test.rdata") 

set.seed(123)
results.bagged.tree <- bagging(purchased ~ gender + age + city 
                       + socioeconomic_status 
                       + sum_of_previous_purchases 
                       + number_of_family_members_who_purchased, 
                       data = d.training, 
                       mfinal = 10) #number of bootstrap samples

# Prediction - Class Membership
bagged.tree.predictions <- predict(object = results.bagged.tree,
                                   newdata = d.test) 


# 2. Performance evaluation (same as above)
# ------------------------------------------------------------------------------
tab.knn.bagged <- table(bagged.tree.predictions$class, d.test$purchased,
                        dnn = c("pred", "actual"))

confusionMatrix(tab.knn.bagged, positive = "1")

# Return detailed information on bagging procedure
results.bagged.tree[["votes"]]
results.bagged.tree[["prob"]][, 2]
results.bagged.tree[["class"]]


