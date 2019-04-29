library(readr)
library(tidyverse)
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

# replace unconvinient columnname characters
names(votedata) <- gsub("-", "_", names(votedata))

str(votedata)

par(mfrow = c(2,3))
for (col in colnames(votedata)[2:7]) {
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
op <- par(mar=c(13,4,4,2)) 
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


# try more parameters

accs <- list()

for (k in seq(2, 16)) {
  model <- randomForest(class ~ ., data = train, mtry = k)
  pred <- predict(model, valid, type = "class")
  cm <- confusionMatrix(pred, valid$class)
  accs <- c(accs, cm$overall[1])
}
plot(2:16, accs)

library(MASS)
step.model <- stepAIC(model_rf_2, direction = "both", trace = FALSE)

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
#library(klaR)
#stepclass(class~., data=train, method="naiveBayes", grouping=c("democrat", "republican")) 

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
for (k in 2:40) {
  pred_knn <- knn(train_numeric[,2:17], valid_numeric[,2:17], train_numeric$class,k=k)
  cm <- confusionMatrix(pred_knn, valid_numeric$class)
  accs <- c(accs, cm$overall[1])
}
par(mfrow = c(1,1))
accs
plot(2:40, accs)
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
pred3 <- predict(model5, test, type = "raw")

result <- cbind(test$ID, as.character(pred3))
colnames(result) <- c("ID", "class")
write.table(result, file="res1.csv", col.names = TRUE, row.names = FALSE, sep = ",",quote=FALSE)