
#####Loading the data set into R ########


setwd("C:\\Users\\Anusha Narthu\\Documents\\R\\595_Project")
Diabetes_original <- read.csv("diabetes_data_upload.csv", sep=",", header = TRUE)

#####converting all variables except Age to factors with two levels

columns <- c(2:17)
Diabetes_original[,columns] <- lapply(Diabetes_original[,columns],factor)

###One hot encoding of all factor columns######
library(mltools)
library(data.table)
newdata <- one_hot(as.data.table(Diabetes_original),dropCols = TRUE)
df <- as.data.frame(newdata)
df[,2:33] <- lapply(df[,2:33],factor)
df[ ,c(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32)] <- list(NULL)
install.packages('tidyverse')
library(tidyverse)
df %>% 
  rename(
    Gender =  Gender_Male,
    class = class_Positive
  )

######creating data report for Exploratoty Analysis######

library(DataExplorer)
create_report(df)

######Using Naive Bayes Classifier#####


smp_size <- floor(0.70 * nrow(df))  ####using 70-30 train test split
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
training_set <- df[train_ind, ] 
testing_set <- df[-train_ind, ]
x_train <- training_set[,-17]
x_train
y_train <- training_set[,17]
y_train
x_test <- testing_set[, -17]
x_test
y_test <- testing_set[,17]
y_test

####The Naive Bayes Model######
library(e1071)
Naive_Bayes_Model=naiveBayes(y_train ~., data=x_train)
NB_Predictions=predict(Naive_Bayes_Model,as.data.frame(x_test)) 
library(ModelMetrics)
library(caret)
conf_matrix=confusionMatrix(NB_Predictions,y_test)
conf_matrix$byClass

######Plotting the ROC######

install.packages('ROCit')
library(ROCit)
ROCit_obj <- rocit(score=as.numeric(NB_Predictions),class=y_test)
plot(ROCit_obj)
ROCit_obj

install.packages('pROC')
library(pROC)
pROC_obj <- roc(y_test,as.numeric(NB_Predictions),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")
## Warning in plot.ci.se(sens.ci, type = "shape", col = "lightblue"): Low
## definition shape.
plot(sens.ci, type="bars")



########Now, running Naive Bayes Classifier using Cross Validation and evaluating model performance######
library("klaR")

model_nb_cv = train(x_train,y_train,'nb',trControl=trainControl(method='repeatedcv',number=15))
model_nb_cv
Predictions_cv=predict(model_nb_cv,as.data.frame(x_test)) 
conf_matrix=confusionMatrix(Predictions_cv,y_test)
conf_matrix$byClass
ROCit_obj <- rocit(score=as.numeric(Predictions_cv),class=y_test)
plot(ROCit_obj)
ROCit_obj


#######Using random forest classifier########

library(randomForest)
model_rf=randomForest(y_train ~ ., data = x_train, ntree = 500, mtry = 3, importance = TRUE)

predictions_rf=predict(model_rf,x_test,type="class")
conf_matrix=confusionMatrix(predictions_rf,y_test)
conf_matrix$byClass
ROCit_obj <- rocit(score=as.numeric(predictions_rf),class=y_test)
plot(ROCit_obj)
ROCit_obj
######using Random Forest Classifier, incorporating Cross Validation#####
model_rf_cv = train(x_train,y_train,'rf',trControl=trainControl(method='cv',number=15))
Predictions_cv_rf=predict(model_rf_cv,as.data.frame(x_test)) 
ROCit_obj <- rocit(score=as.numeric(Predictions_cv_rf),class=y_test)
plot(ROCit_obj)
ROCit_obj

######using Logistic Regression########
logist_model <- glm(y_train ~., data = x_train, family = "binomial")
logist_predictions=predict(logist_model,as.data.frame(x_test))
logist_predictions
logist_predictions <- ifelse(logist_predictions > 0.8, 1, 0)
ROCit_obj <- rocit(score=as.numeric(logist_predictions),class=y_test)
plot(ROCit_obj)
ROCit_obj

#####logistic regression using cross validation#####
model_lm_cv = train(x_train,y_train,'glm',trControl=trainControl(method='cv',number=5))
Predictions_cv_lm=predict(model_lm_cv,as.data.frame(x_test)) 
ROCit_obj <- rocit(score=as.numeric(Predictions_cv_lm),class=y_test)
plot(ROCit_obj)
ROCit_obj


########Feature importance########

library(mlbench)
library(caret)
library(data.table)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(class_Positive~., data=df, method="lvq", preProcess="scale", trControl=control)
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

###### Running Naive Bayes Classifier using top 5 features and evaluating the model again#######
Naive_Bayes_Model_Top5=naiveBayes(y_train ~ Polyuria_Yes + Polydipsia_Yes + Gender_Male + sudden.weight.loss_Yes + partial.paresis_Yes, data=x_train)
NB_PredictionTop5=predict(Naive_Bayes_Model_Top5,as.data.frame(x_test)) 
library(ModelMetrics)
library(caret)
conf_matrix=confusionMatrix(NB_PredictionTop5,y_test)
conf_matrix$byClass
ROCit_obj <- rocit(score=as.numeric(NB_PredictionTop5),class=y_test)
plot(ROCit_obj)
ROCit_obj
pROC_obj <- roc(y_test,as.numeric(NB_PredictionTop5),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)
sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")
## Warning in plot.ci.se(sens.ci, type = "shape", col = "lightblue"): Low
## definition shape.
plot(sens.ci, type="bars")


######XGboost####
install.packages('xgboost')
install.packages('readr')
install.packages('stringr')
install.packages('caret')
install.packages('car')
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)


data_train_matrix = data.matrix(x_train)
y_train_matrix = data.matrix(y_train)
data_test_matrix= data.matrix(x_test)

xgb <- xgboost(data = data_train_matrix, 
               label = y_train_matrix, 
               eta = 1,
               max_depth = 2, 
               nround=2, 
               
               objective = "binary:logistic",
               
               nthread = 2)
predictions_xgb <- predict(xgb, data_test_matrix)
predictions_xgb
ROCit_obj <- rocit(score=predictions_xgb,class=y_test)

plot(ROCit_obj)
ROCit_obj

#names <- [[2]]
names

data_train_matrix
importance_matrix <- xgb.importance(data_train_matrix@Dimnames[[2]], model = xgb)
importance_matrix
xgb.plot.importance()

install.packages('kernlab')
install.packages('e1071')
library(kernlab)      # SVM methodology
library(e1071)  
svm_model <- svm(y_train ~ ., data=data.matrix(x_train,rownames.force = NA))
svm_predict <- predict(svm_model, data_test_matrix)
svm_predict
ROCit_obj <- rocit(score=as.numeric(svm_predict),class=y_test)

plot(ROCit_obj)
ROCit_obj
