#importing library 
library(ggplot2)
library(corrplot)
library(caTools)
library(ROCR) 
library(MASS)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(kernlab)
library(readr)
library(ggpubr)

#Clearing the r environment
rm(list = ls(all = TRUE))


#loading data
clg_ad <- read.csv("College_admission.csv")
head(clg_ad)
tail(clg_ad)


#descriptive statistics
summary(clg_ad$gpa)


#1. Find the missing values. (if any, perform missing value treatment)

sum(is.null(clg_ad))
#Since there are no Null values in the dataset, no need for missing value treatment


#2. Find outliers (if any, then perform outlier treatment).
#Visualizing continuous variables

hist(clg_ad$gre)
hist(clg_ad$gpa)

#Using Boxplot to understand if there are any outliers present in GRE variable
boxplot(clg_ad$gre, horizontal = T)
# By looking at the Boxplot, we can see there are outliers in GRE variable
greoutlier <- boxplot(clg_ad$gre)$out

#checking length of GRE outliers
length(greoutlier)

#Removing outliers from GRE variable
Q <- quantile(clg_ad$gre, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(clg_ad$gre)

up <- Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range

clg_ad <- subset(clg_ad, clg_ad$gre > (Q[1] - 1.5*iqr) & clg_ad$gre < (Q[2]+1.5*iqr))
head(clg_ad)


#Using Boxplot to understand if there are any outliers present in GPA variable
boxplot(clg_ad$gpa, horizontal = T)
# By looking at the Boxplot, we can see there are outliers in GPA variable
gpaoutlier <- boxplot(clg_ad$gpa)$out

length(gpaoutlier)
#There is only one outlier present in this Variable, we can remove it.

#Removing outliers from GPA variable
Q <- quantile(clg_ad$gpa, probs=c(.25, .75), na.rm = FALSE)

iqr <- IQR(clg_ad$gpa)

up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range

clg_ad <- subset(clg_ad, clg_ad$gpa > (Q[1] - 1.5*iqr) & clg_ad$gpa < (Q[2]+1.5*iqr))
head(clg_ad)

#now outliers have been removed from data, After removing outliers we have 395 data points

#3. Find the structure of the data set and if required, transform the numeric data type to factor and vice-versa.
#structure of data
str(clg_ad)

clg_ad$admit <- as.factor(clg_ad$admit)
clg_ad$ses <- as.factor(clg_ad$ses)
clg_ad$Gender_Male <- as.factor(clg_ad$Gender_Male)
clg_ad$Race <- as.factor(clg_ad$Race)
clg_ad$rank <- as.factor(clg_ad$rank)
str(clg_ad)

#4. Find whether the data is normally distributed or not. Use the plot to determine the same.

#checking distribution of GRE variable
plot(density(clg_ad$gre))

ggqqplot(clg_ad$gre)

#Since Majority of the data points fall on the line,
#we can assume that the data is normally distributed

##Normality test using Shapiro test
shapiro.test(clg_ad$gre)

#checking distribution of GPA variable
plot(density(clg_ad$gpa))

##Since Majority of the data points fall on the line, we can assume that the data is normally distributed
ggqqplot(clg_ad$gpa)

#Normality test using Shapiro test
shapiro.test(clg_ad$gpa)

#5. Normalize the data if not normally distributed
#Since the data is normally distributed, data tranformation is not required
#But however there is variance in GRE and GPA variable, we can clearly see
  #that gre values are in hundred times more than the gpa values.
#In this case when we build model, gre will internally influence 
  #the result more due to its larger value.
#To avoid problem and for accurate model,
  #we can scale down the variables to avoid problem and accurate model.

#Creating a copy of the data
clgad <- clg_ad
clg_ad$gre <- scale(clg_ad$gre, center = T, scale = T)
clg_ad$gpa <- scale(clg_ad$gpa, center = T, scale = T)
head(clg_ad)

#6. Use variable reduction techniques to identify significant variables.

#We can build logistic regression model and identify significant variables.
set.seed(1234)
sampledata <- sample.split(clg_ad$admit, SplitRatio = 0.7)
train <- clg_ad[sampledata==T,]
test <- clg_ad[sampledata==F,]

test_without_admit <- test[,-1]

# fit the model
class(train$admit)

log_reg <- glm(admit ~ . , data = train, family = 'binomial')
summary(log_reg)

#Using step AIC method to identify significant variables
model_AIC = stepAIC(object = log_reg,direction = "both")
summary(model_AIC)

#From the results above GRE, GPA and Rank are the most significant variables.


#7. Run logistic model to determine the factors that 
    #influence the admission process of a student (Drop insignificant variables)
summary(log_reg)

#After looking at the summary results obtained from logistic regression, 
  #factors that influence admission processs are GRE, GPA and RANK

# Dropping variables that are insignificant.
clg_ad1 <- subset(clg_ad, select = -c(ses,Gender_Male,Race))
head(clg_ad1)

#8. Calculate the accuracy of the model and run validation techniques
#Using the result of logistic regresssion model to calculate the accuracy
#Using only the significant variables for Building models
class(clg_ad1)
head(clg_ad1)

set.seed(123)

sample_data <- sample.split(clg_ad1$admit, SplitRatio = 0.7)
Train <- clg_ad1[sample_data==T,]
Test <- clg_ad1[sample_data==F,]

test_without_admit <- Test[,-1]

logreg_model <- glm(admit ~ . , data = Train, family = 'binomial')

summary(logreg_model)


#Predicting on test data and calculating the accuracy using confusion matrix
## Model Evaluation
prob_train <- predict(logreg_model,newdata = Test[,-1], type = "response")
preds_train <- ifelse(prob_train > 0.49,1,0) # use 0.5 or 0.49 to get the best accuracy
comp = table(Test$admit,preds_train)
confusionMatrix(comp,positive = "0")


#From the above model we can see that
  #logistic regression model is predicting the admission rate is 73.11%



#9. Try other modelling techniques like decision tree and SVM and select a champion model

#Decision Tree
#Using library C50
library(C50)


#Building the model and printing the summary
c5_tree_model <- C5.0(admit~., Train, rules = T)
c5_tree_model

summary(c5_tree_model)

#Predicting on test data
prob_pred <- predict(c5_tree_model, Test[,-1])
prob_pred


#Using confusion metric to calculate accuracy
confusionMatrix(prob_pred,Test$admit)

#using Decision Tree (library C50) is giving 68.07% accuracy

#Decision Tree using rpart
#Model Building
rpart_tree_model <- rpart(admit ~ .,
                    data = Train,
                    method = "class")


# display decision tree
prp(rpart_tree_model)

# make predictions on the test set
tree_predict <- predict(rpart_tree_model, Test[,-1], type = "class")
tree_predict

# evaluate the results
confusionMatrix(tree_predict, as.factor(Test$admit), positive = "0")  

#using Decision Tree (rpart) model is giving 75.63% accuracy

### Build a random forest 
rf_model <- randomForest(admit ~ ., data=Train, proximity=FALSE,
                         ntree=15, mtry=3, na.action=na.omit)
rf_model
summary(rf_model)

#using the model to predict on test data and using confusion matrix to calculate accuracy
rf_pred <- predict(rf_model, newdata=Test[,-1], type='Class')

confusionMatrix(rf_pred, Test$admit, positive = "0")
#using Random Forrest model is giving 68.07% accuracy

# build the Support Vector Machines(SVM) model
svm_model<- ksvm(admit ~ ., data = Train,scale = FALSE , C=25)
summary(svm_model)

# Predicting the model results 
svm_predict <- predict(svm_model, Test[,-1])

#SVM model accuracy using confusion matrix
confusionMatrix(svm_predict, Test$admit)
#using SVM model is giving 69.75% accuracy


# 10. Determine the accuracy rates for each kind of model
# a) Logistic Regression : 73.11%
# b) Decision Tree (C50) : 68.07%
# c) Decision Tree (rpart) : 75.63%
# d) Random Forest : 68.07%
# e) SVM : 69.75%


# 11. Select the most accurate model
# Decision - Decision Tree using library rpart is giving best accuracy 
 # which is 75.63% compared to other models
# So Most accurate model is Decision Tree(Rpart) Model


# 12. Identify other Machine learning or statistical techniques
# I have already used Random Forest. Other than random forest,
# we can apply Naive Bayes and Boosting Algorithms.

# Naive Bayes
library(naivebayes)
naive_bayes_model <- naive_bayes(admit~.,Train)
naive_bayes_model

pred_nb <- predict(naive_bayes_model,Test[,-1])
confusionMatrix(pred_nb,Test$admit)

train.control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
model_xgbTree <- train(admit ~ .,data=Train, method = "xgbTree",
                       trControl = train.control, verbosity = 0)
print(model_xgbTree)

pred_xgboost <- predict(model_xgbTree,Test[,-1])
confusionMatrix(pred_xgboost,Test$admit)


# Descriptive:
#   Categorize the average of grade point into High, Medium,
#and Low (with admission probability percentages) and plot it on a point chart.
# Cross grid for admission variables with GRE Categorization is shown below:
  
# GRE Categorized 0-440 Low 440-580 Medium 580+ High

head(clgad)

clgad$Category[clgad$gre < 440] <- "Low"
clgad$Category[clgad$gre >= 440 & clgad$gre < 580] <- "Medium"
clgad$Category[clgad$gre >= 580] <- "High"
head(clgad)


clgad$Category <- as.factor(clgad$Category)
str(clgad)

summary(clgad$Category)


plot(clgad$Category)

