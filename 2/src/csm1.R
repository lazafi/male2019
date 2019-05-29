library(readxl)
# Function for Root Mean Squared Error
RMSE <- function(res) { 
  RSS <- c(crossprod(res$residuals))
  MSE <- RSS / length(res$residuals)
  RMSE <- sqrt(MSE)
  RMSE
}



url <- "2/data/CSM/2014 and 2015 CSM dataset.xlsx"

odata <- read_excel(url)
# rename columns
colnames(odata)[colnames(odata)=="Aggregate Followers"] <- "Followers"
str(odata)
summary(odata)

data <- odata

# data analysis
hist(data$Ratings)

hist(log(data$Gross))

pairs(~Ratings+Budget+Screens+Likes+Dislikes,data=data, main="Simple Scatterplot Matrix")
# print rows with missing values
data[!complete.cases(data),]

# use linear regression to fill in missing values

data_nona <- na.omit(data)

##Agregate Followers
### number of nans
sum(is.na(odata$Followers))

# print rows with missing values
odata[!complete.cases(data),14]

m_flw <- lm(Followers~., data=data_nona[,2:14])
### eval predictions
plot(data_nona$Followers, predict(m_flw))
abline(c(0,1), col="red")
#### not good, use median instead to fill in missing values
data[which(is.na(data$Followers)),14] <- median(data$Followers, na.rm = TRUE)


## Screens
### number of nans
sum(is.na(odata$Screens))

m_sc1 <- lm(Screens~., data=data_nona[,2:14])


### eval predictions
plot(data_nona$Screens, predict(m_sc1))
abline(c(0,1), col="red")
#### good enought 
filldata_screens <- data[which(is.na(data$Screens)),]
data[which(is.na(data$Screens)),7] <- predict(m_sc1, newdata=filldata_screens[,2:14])


##Budget
### number of nans
sum(is.na(odata$Budget))

m_bud <- lm(Budget~., data=data_nona[,2:14])

plot(data_nona$Budget, predict(m_bud))
abline(c(0,1), col="red")
## good enough

data[which(is.na(data$Budget)),6] <- predict(m_bud, newdata=data[which(is.na(data$Budget)),2:14])

data[!complete.cases(data),]
### all na filled!



# regression models
library(caret)
set.seed(123)
#omp_set_num_threads(4) # caret parallel processing threads

# use cv to train models
train_control <- trainControl(
  method="cv", 
  number=10,  
  verboseIter = TRUE,
  allowParallel = TRUE # FALSE for reproducible results 
)


# train lm model, use it as baseline
formula <- as.formula("Gross~Ratings+Genre+Budget+Screens+Sequel+Sentiment+Views+Likes+Dislikes+Comments+Followers")

#model <- train(formula, data=data, trControl=train_control, method = 'lm')
model <- lm(formula, data)
summary(model)
RMSE(model)

plot(data$Gross, predict(model$finalModel))
abline(c(0,1), col="red")

# try with centered and scaled values
data_cs <- data.frame(scale(data[2:14], TRUE, TRUE))
model2 <- lm(formula, data=data_cs)
#model2 <- train(formula, data=data, trControl=train_control, method = 'lm', preProcess = c("center", "scale"))
summary(model2)
RMSE(model2)

# prediction figure with bad predictions hightlighted
ra <- abs(data$Gross - predict(model2$finalModel))
table <- data.frame(cbind(ra, data$Gross, predict(model2$finalModel), model2$finalModel$residuals))
table$Movie <- data$Movie
most <- table[order(table[,1], decreasing = T),][1:10,]
plot(data$Gross, predict(model2$finalModel), xlab="Gross", ylab="predicted Gross", main="Least Square Regression Error")
abline(c(0,1), col="red")
text(x=most[,2], y=most[,3]+0.16e+08, labels=most[,5], cex=1.3)
points(x=most[,2], y=most[,3], col="red")

#residual plot
plot(predict(model2$finalModel), model2$finalModel$residuals, xlab="predicted Gross", ylab="residual")
most <- table[order(table[,4], decreasing = T),][1:10,]
text(x=most[,4], y=most[,2], labels=most[,5], cex=1)



# robust regression
model_rob <- train(formula, data=data, trControl=train_control, method = 'rlm',  preProcess = c("center", "scale"))
model_rob$results

plot(data$Gross, predict(model_rob$finalModel), pch = model_rob$finalModel$weights)
abline(c(0,1), col="red")

# random forrest

grid_rf <- expand.grid(
#  mtry = c(1, 10, 20, 50, 100)
  mtry = c(1, 2, 5, 10, 20, 50, 100)
)

model_rf <- train(
  formula,
  data=data,
  trControl = train_control,
  tuneGrid = grid_rf,
  verbose = TRUE,
  method="rf",
  metric="RMSE",
  preProcess = c("center", "scale")
  
)

# plot mtry param tuning
#plot(model_rf$results$mtry, model_rf$results$Rsquared, xlab="mtry", ylab="Rsquared", log = "x", type = "b", main = "Random Forests Parameter Tuning")
plot(model_rf$results$mtry, model_rf$results$RMSE, xlab="mtry", ylab="RMSE", log = "x", type = "b", main = "Random Forests Parameter Tuning")

plot(data$Gross, predict(model_rf$finalModel), xlab="y", ylab="y-hat")
abline(c(0,1), col="red")

#residual plot
#plot(predict(model_rf$finalModel), model_rf$finalModel$residuals, xlab="predicted Gross", ylab="residual", main="Random Forest Residual Plot")

#ridge
#######

#getModelInfo("glmnet")$glmnet$parameters
#parameter   class                    label
#1     alpha numeric        Mixing Percentage
#2    lambda numeric Regularization Parameter

#lambdas <- 10^seq(3, -2, by = -.1)
grid_ridge <- expand.grid(
  alpha = 0,
  lambda = 10^seq(-3, 5, length.out = 100)
)

model_ridge <- train(
  formula,
  data=data,
  trControl = train_control,
  tuneGrid = grid_ridge,
  method="glmnet"
)

model_ridge$results
plot(model_ridge$results$lambda, model_ridge$results$Rsquared, xlab="lamdba", ylab="r²", type = "l")
abline(v=model_ridge$finalModel$lambdaOpt, col="red")

#######
#######

#getModelInfo("glmnet")$glmnet$parameters
#parameter   class                    label
#1     alpha numeric        Mixing Percentage
#2    lambda numeric Regularization Parameter

grid_lasso <- expand.grid(
  alpha = c(0, 0.25, 0.5, 0.75, 1),
  lambda = c(0, 0.25, 0.5, 0.75, 1)
)

model_lasso <- train(
  formula,
  data=data,
  trControl = train_control,
  tuneGrid = grid_lasso,
  method="glmnet"
)

model_lasso$results

plot(model_lasso$results$alpha, model_lasso$results$RMSE, xlab="alpha", ylab="RMSE", type = "l")
plot(model_lasso$results$lambda, model_lasso$results$RMSE, xlab="lambda", ylab="RMSE", type = "l")
plot(model_lasso$results$alpha, model_lasso$results$RSquare, xlab="alpha", ylab="RSquare", type = "l")
plot(model_lasso$results$lambda, model_lasso$results$RSquare, xlab="lambda", ylab="RSquare", type = "l")

#######
#knn

#getModelInfo("knn")$knn$parameters
#parameter   class      label
#1         k numeric #Neighbors

grid_knn <- expand.grid(
  k = seq(2, 50, by=5)
)

model_knn <- train(
  formula,
  data=data,
  trControl = train_control,
  tuneGrid = grid_knn,
  preProcess = c("center", "scale"),
  method="knn",
  metric="RMSE"
)

model_knn$results
plot(model_knn$results$k, model_knn$results$RMSE, xlab="k", ylab="RMSE", type = "l", main="kNN Tuning k-parameter")
abline(v=model_knn$finalModel$k, col="red")
axis(side=1, at=model_knn$finalModel$k, labels=model_knn$finalModel$k)

##SVM
#> getModelInfo("svmRadial")$svmRadial$parameters
#parameter   class label
#1     sigma numeric Sigma
#2         C numeric  Cost

grid_svm <- expand.grid(
  sigma = seq(0.001, 0.02, by=0.01),
  C = c(1.5,1.596,1.65,1.89,1.95,2,2.2,2.44)
)

model_svm <- train(
  formula,
  data=data,
  trControl = train_control,
  tuneGrid = grid_svm,
  preProcess = c("center", "scale"),
  method="svmRadial"
)

model_svm$results
plot(model_svm$results$sigma, model_svm$results$Rsquared, xlab="k", ylab="r²", type = "l")
plot(model_svm$results$C, model_svm$results$Rsquared, xlab="k", ylab="r²", type = "l")

