library(readr)
library(tidyverse)
library(caret)
library(e1071)
# col_types = cols(ID = col_skip())
url <- "../data/CongressionalVoting/CongressionalVotingID.shuf.train.csv"

votedata <- read_csv(url, col_types = cols(
  ID = col_skip(),
  class = col_factor(NULL),
  `handicapped-infants` = col_factor(c("y","n","unknown")),
  `water-project-cost-sharing` = col_factor(c("y","n","unknown")),
  `adoption-of-the-budget-resolution` =col_factor(c("y","n","unknown")),
  `physician-fee-freeze` = col_factor(c("y","n","unknown")),
  `el-salvador-aid` = col_factor(c("y","n","unknown")),
  `religious-groups-in-schools` = col_factor(c("y","n","unknown")),
  `anti-satellite-test-ban` = col_factor(c("y","n","unknown")),
  `aid-to-nicaraguan-contras` = col_factor(c("y","n","unknown")),
  `mx-missile` = col_factor(c("y","n","unknown")),
  immigration = col_factor(c("y","n","unknown")),
  `synfuels-crporation-cutback` = col_factor(c("y","n","unknown")),
  `education-spending` = col_factor(c("y","n","unknown")),
  `superfund-right-to-sue` = col_factor(c("y","n","unknown")),
  crime = col_factor(c("y","n","unknown")),
  `duty-free-exports` = col_factor(c("y","n","unknown")),
  `export-administration-act-south-africa` = col_factor(c("y","n","unknown"))
))

votedata_na <- read_csv(url, col_types = cols(
  ID = col_skip(),
  class = col_factor(NULL),
  `handicapped-infants` = col_factor(c("y","n")),
  `water-project-cost-sharing` = col_factor(c("y","n")),
  `adoption-of-the-budget-resolution` =col_factor(c("y","n")),
  `physician-fee-freeze` = col_factor(c("y","n")),
  `el-salvador-aid` = col_factor(c("y","n")),
  `religious-groups-in-schools` = col_factor(c("y","n")),
  `anti-satellite-test-ban` = col_factor(c("y","n")),
  `aid-to-nicaraguan-contras` = col_factor(c("y","n")),
  `mx-missile` = col_factor(c("y","n")),
  immigration = col_factor(c("y","n")),
  `synfuels-crporation-cutback` = col_factor(c("y","n")),
  `education-spending` = col_factor(c("y","n")),
  `superfund-right-to-sue` = col_factor(c("y","n")),
  crime = col_factor(c("y","n")),
  `duty-free-exports` = col_factor(c("y","n")),
  `export-administration-act-south-africa` = col_factor(c("y","n"))
),  na = "unknown")

# replace unconvinient columnname characters
names(votedata) <- gsub("-", "_", names(votedata))
names(votedata_na) <- gsub("-", "_", names(votedata_na))

str(votedata)

par(mfrow = c(2,3), mar=c(2,4,4,2))
for (col in colnames(votedata)[c(5, 4, 6, 13, 9, 12)]) {
  print(col)
  hit <- table(votedata$class, votedata[[col]])
  mosaicplot(hit, col=c(3,2,1), main=col)
}
for (col in colnames(votedata)[8:13]) {
  print(col)
  hit <- table(votedata$class, votedata[[col]])
  mosaicplot(hit, col=c(3,2,1), main=col)
}
for (col in colnames(votedata)[14:17]) {
  print(col)
  hit <- table(votedata$class, votedata[[col]])
  mosaicplot(hit, col=c(3,2,1), main=col)
}



# use random forests to calculate importance
library(randomForest)
model_rf <- randomForest(class ~ ., data = votedata, importance = TRUE)
ginis <- model_rf$importance[,4]
op <- par(mar=c(2,14,4,2)) 
barplot(ginis[order(ginis)], horiz = TRUE, las=1, main= "Importance", xlab = "Gini",cex.axis=0.8, cex.names=0.8)
rm(op)

# split validataion

set.seed(123)
randsample <- sample(nrow(votedata), 0.8*nrow(votedata), replace = FALSE)
train <- votedata[randsample,]
valid <- votedata[-randsample,]

# baseline random classifier 
baseline <- sample(c("democrat", "republican"), nrow(valid), replace = TRUE)
#baseline always democrat
baseline <- rep("democrat", nrow(valid))
confusionMatrix(as.factor(baseline), valid$class)

# random forests
#install.packages("randomForest")
library(randomForest)
model_rf_1 <- randomForest(class ~ ., data = train, importance = TRUE)

pred <- predict(model_rf_1, valid, type = "class")
table(pred, valid$class)
confusionMatrix(pred, valid$class)

model_rf_2 <- randomForest(class ~ ., data = train, ntree = 100)
pred <- predict(model_rf_2, valid, type = "class")
confusionMatrix(pred, valid$class)

#ommit physician-fee-freese

model_rf_3 <- randomForest(class ~ . - physician_fee_freeze, data = train)
confusionMatrix(predict(model_rf_3, valid, type = "class"), valid$class)

# try with na
train_na <- votedata_na[randsample,]
valid_na <- votedata_na[-randsample,]

model_rf_na_1 <- randomForest(class ~ ., data = train_na , na.action=na.omit)
confusionMatrix(predict(model_rf_na_1, valid_na, type = "class"), valid_na$class)


# try more parameters

accs <- list()

for (k in seq(2, 16)) {
  model <- randomForest(class ~ ., data = train, mtry = k)
  pred <- predict(model, valid, type = "class")
  cm <- confusionMatrix(pred, valid$class)
  accs <- c(accs, cm$overall[1])
}
plot(2:16, accs)

getTree(model_rf_1, labelVar = TRUE)

#library(MASS)
#step.model <- stepAIC(model_rf_2, direction = "both", trace = FALSE)

# cross validation
#library(cvTools)

#rawdata <- read_csv(url) 
## replace unconvinient columnname characters
#names(rawdata) <- gsub("-", "_", names(rawdata))
#str(rawdata)
#model2 <- randomForest(class ~ ., data = votedata)
#m2_cv <- cvFit(model2, data = votedata, y = class)


#library(caret)
#train_control <- trainControl(method="cv", number=10)
#grid <- expand.grid(mtry=c(2,5,10))
#grid
#model3 <- train(class~., data=votedata, trControl=train_control, method="rf", tuneGrid=grid)
#model3

# split naive bayes
model4 <- naiveBayes(class ~., data=train)
model4

pred4 <- predict(model4, valid, type = "class")
table(pred4, valid$class)
confusionMatrix(pred4, valid$class)
#library(MASS)
#step.model <- stepAIC(model4, direction = "both", trace = FALSE)
library(klaR)
stepclass(train[,2:17], t(train[,1]), "naiveBayes", start.vars = "physician_fee_freeze") 
#stepclass(formula=class~., data=train, method="naiveBayes") 

stepclass(train[,2:17], t(train[,1]), "naiveBayes", direction = "backward") 
stepAIC(model4, class~.)


#use only a subset
accs <- list()
for (i in 1:16) {
  print(formula)
  print(paste(names(ginis[order(-ginis)][2:i]), collapse = ' '))
  formula <- paste("class ~", paste(names(ginis[order(-ginis)][2:i]), collapse = '+'))
  model_nb_2 <- naiveBayes(as.formula(formula), data=train)
  cm <- confusionMatrix(predict(model_nb_2, valid, type = "class"), valid$class)
  print(cm$overall[1])
  accs = c(accs, cm$overall[1])
}
par(mfrow = c(1,1))
plot(1:16, accs, xlab="features", ylab="accuracy")

# model for kaggle
model_nb_3 <- naiveBayes(class ~ adoption_of_the_budget_resolution+education_spending+el_salvador_aid+aid_to_nicaraguan_contras+mx_missile+synfuels_crporation_cutback, data=train)

# cv naive bayes
train_control <- trainControl(method="cv", number=10)
grid <- expand.grid(fL=c(0), usekernel=c(FALSE), adjust=c(TRUE))
grid
model5 <- train(class~.-ID, data=rawdata, trControl=train_control, method="nb", tuneGrid=grid)
model5

# split knn

#convert to numeric
train_numeric <- train
train_numeric[,2:17] <- lapply(train[,2:17], function(x) as.numeric(x) );
valid_numeric <- valid
valid_numeric[,2:17] <- lapply(valid[,2:17], function(x) as.numeric(x) );

library(class)

accs <- list()
for (k in 2:16) {
  pred_knn <- knn(train_numeric[,2:17], valid_numeric[,2:17], train_numeric$class,k=k)
  cm <- confusionMatrix(pred_knn, valid_numeric$class)
  accs <- c(accs, cm$overall[1])
}
par(mfrow = c(1,1))
accs
plot(2:10, accs)
# predict kaggle test set
testurl <- "../data/CongressionalVoting/CongressionalVotingID.shuf.test.csv"
test <- read_csv(testurl, col_types = cols( 
                                             `handicapped-infants` = col_factor(c("y","n","unknown")),
                                             `water-project-cost-sharing` = col_factor(c("y","n","unknown")),
                                             `adoption-of-the-budget-resolution` =col_factor(c("y","n","unknown")),
                                             `physician-fee-freeze` = col_factor(c("y","n","unknown")),
                                             `el-salvador-aid` = col_factor(c("y","n","unknown")),
                                             `religious-groups-in-schools` = col_factor(c("y","n","unknown")),
                                             `anti-satellite-test-ban` = col_factor(c("y","n","unknown")),
                                             `aid-to-nicaraguan-contras` = col_factor(c("y","n","unknown")),
                                             `mx-missile` = col_factor(c("y","n","unknown")),
                                             immigration = col_factor(c("y","n","unknown")),
                                             `synfuels-crporation-cutback` = col_factor(c("y","n","unknown")),
                                             `education-spending` = col_factor(c("y","n","unknown")),
                                             `superfund-right-to-sue` = col_factor(c("y","n","unknown")),
                                             crime = col_factor(c("y","n","unknown")),
                                             `duty-free-exports` = col_factor(c("y","n","unknown")),
                                             `export-administration-act-south-africa` = col_factor(c("y","n","unknown"))
))

#test <- read_csv(testurl)
names(test) <- gsub("-", "_", names(test))
model12 <- randomForest(class ~ ., data = train, mtry = 12)
#pred3 <- predict(model5, test, type = "class")
pred3 <- predict(model_nb_3, test, type = "class")

result <- cbind(test$ID, as.character(pred3))
colnames(result) <- c("ID", "class")
write.table(result, file="res1.csv", col.names = TRUE, row.names = FALSE, sep = ",",quote=FALSE)