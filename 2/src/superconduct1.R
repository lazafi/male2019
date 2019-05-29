library(readr)
url <- "2/data/superconduct/train.csv"

data1 <- read_csv(url)
str(data1)
summary(data1)

url2 <- "2/data/superconduct/unique_m.csv"
data2 <- read_csv(url2)

# replace all na-s with 0
data2[is.na(data2)] <- 0

# show missing data
data1[!complete.cases(data1),]

# combine data
#data <- cbind(data1, data2[,1:86])
data <- data1

#add info from chemical formula
#formula_data <- read_csv(url2)
#data <- cbind(data, formula_data)

# data analysis

ggplot(data, aes(x=critical_temp)) + 
  geom_histogram(aes(y=..density..), colour="black", bins = 14, fill="white")+
  geom_density(alpha=.2, fill="#FF6666")


##stepwise feature selection
## linear regression with all variables
#
m1 <- lm(critical_temp~.,data=data)
summary(m1)
plot(m1)
print(m1)
#plot(data$critical_temp, predict(m1), xlab="y", ylab="y-hat")
#abline(c(0,1), col="red")

# use stepwise feature selection based on AIC
#m1s <- step(m1)
#summary(m1s)

# dropped variables
#m1s$anova

#plot(data$critical_temp, predict(m1s), xlab="y", ylab="y-hat")
#abline(c(0,1), col="red")


#robust regression

library(robustbase)

# lts
#m2 <- ltsReg(critical_temp~.,data=data)

#summary(m2)
#plot(m2)

# lmrob

#m3 <- lmrob(critical_temp~.,data=data)
#summary(m3)
#plot(m3)


library(caret)
set.seed(123)
#omp_set_num_threads(4) # caret parallel processing threads

# use cv to train models
train_control <- trainControl(
  method = "cv",
  number = 2,
  verboseIter = TRUE, 
  allowParallel = TRUE # FALSE for reproducible results 
)

# train lm model, use it as baseline
model <- train(critical_temp~., data=data, trControl=train_control, method = 'lm')
print(model)

# same model with scale and center data
model2 <- train(critical_temp~., data=data, trControl=train_control, preProcess = c("center", "scale"), method = 'lm')
print(model2)

plot(model2)

#robust reg rlm
#getModelInfo("rlm")

model_rlm <- train(critical_temp~., data=data, trControl=train_control, grid = NA, method = 'rlm')
print(model_rlm)

plot(data$critical_temp, predict(model_rlm$finalModel), xlab="y", ylab="y-hat", pch = model_rlm$finalModel$weights)
abline(c(0,1), col="red")

# residual plot
plot(predict(model_rlm$finalModel), model_rlm$finalModel$residuals, xlab="predicted Gross", ylab="residual")


# random forrest
#$rfRules$parameters
#parameter   class                         label
#1      mtry numeric #Randomly Selected Predictors
#2  maxdepth numeric            Maximum Rule Depth

grid_rf <- expand.grid(
  mtry = c(1, 10, 20)
)

model_rf <- train(
  critical_temp~.,
  data=data,
  trControl = train_control,
  tuneGrid = grid_rf,
  verbose = TRUE,
  na.action=na.omit
)

plot(model_rf$results$mtry, model_rf$results$Rsquared, xlab="mtry", ylab="r²", log = "x", type = "b")

plot(data$critical_temp, predict(model_rf$finalModel), xlab="y", ylab="y-hat")
abline(c(0,1), col="red")



# xgboost
# xgboost is used in the paper atached to the original data
#https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret


#$xgbTree$parameters
#parameter   class                          label
#1          nrounds numeric          # Boosting Iterations
#2        max_depth numeric                 Max Tree Depth
#3              eta numeric                      Shrinkage
#4            gamma numeric         Minimum Loss Reduction
#5 colsample_bytree numeric     Subsample Ratio of Columns
#6 min_child_weight numeric Minimum Sum of Instance Weight
#7        subsample numeric           Subsample Percentage


grid_default <- expand.grid(
#  mtry = 100,
  nrounds = c(100,200),
  max_depth = 15:25,
  eta = c(00.1, 0.015, 0.020),
#  gamma = 0,
  colsample_bytree = c(0.25, 0.5, 0.75),
  min_child_weight = c(1, 10),
  subsample = 0.5
)


xgb_base <- train(
  critical_temp~.,
  data=data,
  mtry = 100,
  trControl = train_control,
  tuneGrid = grid_default,
  verbose = TRUE,
  na.action=na.omit
)
###ridge

grid_ridge <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-3, 5, length.out = 100)
)

model_ridge <- train(
  critical_temp~.,
  data=data,
  trControl = train_control,
  tuneGrid = grid_ridge,
  method="glmnet"
)

model_ridge$results
x<-model_ridge$results$lambda
y <- model_ridge$results$RMSE
plot(x, y, xlab="lamdba", ylab="r²", type = "p")

plot(model_ridge$results$lambda, model_ridge$results$RMSE, xlab="lamdba", ylab="r²", type = "p")
abline(v=model_ridge$finalModel$lambdaOpt, col="red")
##

#knn

#getModelInfo("knn")$knn$parameters
#parameter   class      label
#1         k numeric #Neighbors

grid_knn <- expand.grid(
  k = seq(2, 102, by=10)
)

model_knn <- train(
  critical_temp~.,
  data=data,
  trControl = train_control,
  tuneGrid = grid_knn,
  method="knn"
)

model_knn$results
plot(model_knn$results$k, model_knn$results$Rsquared, xlab="k", ylab="r²", type = "l")

plot(model_knn$results$k, model_knn$results$RMSE, xlab="k", ylab="RSME", type = "b", main="kNN Parameter Tuning")

model_knn2 <- train(
  critical_temp~.,
  data=data,
  trControl = train_control,
  tuneGrid = grid_knn,
  method="knn",
  preProcess = c("center", "scale")
)

plot(model_knn2$results$k, model_knn2$results$RMSE, xlab="k", ylab="RSME", type = "b", main="kNN Parameter Tuning")

