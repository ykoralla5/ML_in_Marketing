# ------------------------------------------------------------------------------
# Boosting
# ------------------------------------------------------------------------------


# 1. Apply Boosting
# ------------------------------------------------------------------------------

# Load R packages
# install.packages("adabag")
library(adabag)

# Load data
load(file = "d_splitted_in_train_and_test.rdata")

results.boosted.tree <- boosting(purchased ~ gender + age + city
                                 + socioeconomic_status 
                                 + sum_of_previous_purchases 
                                 + number_of_family_members_who_purchased, 
                                 data = d.training, 
                                 mfinal = 10)

# Prediction - Class Membership
boosted.tree.predictions <- predict.boosting(object = results.boosted.tree,
                                             newdata = d.test) 

# 2. Performance evaluation
# ------------------------------------------------------------------------------
tab.boosted.tree <- table(boosted.tree.predictions$class,
                         d.test$purchased,
                         dnn = c("pred", "actual"))
confusionMatrix(tab.boosted.tree, positive = "1")

# Return detailed information on bagging procedure
results.boosted.tree[["votes"]]
results.boosted.tree[["prob"]][, 2]
results.boosted.tree[["class"]]




