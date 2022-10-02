# ------------------------------------------------------------------------------
#  Workflow of supervised machine learning 
# ------------------------------------------------------------------------------


# 1. Load data
# ------------------------------------------------------------------------------
library(data.table)
d <- fread(input = "C:/Users/Yukta/Dropbox/My PC (LAPTOP-T9KDNN4E)/Desktop/Yukta's folder/Switzerland/UZH Information Systems/Semester 2/Machine Learning - a non-technical intro/Data/data_ml_class.csv", header=TRUE)
# d <- read.csv("data_ml_class.csv", sep = ";")


# 2. Check if the data is loaded correctly 
# ------------------------------------------------------------------------------
# (a) Print first 10 lines
head(d)

# (b) Check variable type
str(d)


# 3. Assign right type for categorical variables
# ------------------------------------------------------------------------------
d[, purchased:=as.factor(purchased)] 
d[, gender:=as.factor(gender)] 
d[, city:=as.factor(city)] 
d[, socioeconomic_status:=as.factor(socioeconomic_status)] 


# 4. Get rid of the variables that might have no use in our classifcation task
# ------------------------------------------------------------------------------
d[, name:=NULL] 


# 5. Get summary statistics for dataset
# ------------------------------------------------------------------------------
summary(d)


# 6. Save data frame in binary format
# ------------------------------------------------------------------------------
saveRDS(object = d, file= "d.rds")
