library(readr)
library(tidyverse)
# col_types = cols(ID = col_skip())
url <- "../data/CongressionalVoting/CongressionalVotingID.shuf.train.csv"

data <- read_csv(url, col_types = cols(
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
str(data)

par(mfrow = c(2,3))
for (col in colnames(data)[2:7]) {
  print(col)
  hit <- table(data$class, data[[col]])
  mosaicplot(hit, col=c(3,2,1), main=col)
}
for (col in colnames(data)[8:13]) {
  print(col)
  hit <- table(data$class, data[[col]])
  mosaicplot(hit, col=c(3,2,1), main=col)
}
for (col in colnames(data)[14:17]) {
  print(col)
  hit <- table(data$class, data[[col]])
  mosaicplot(hit, col=c(3,2,1), main=col)
}

set.seed(100)
randsample <- sample(nrow(data), 0.7*nrow(data), replace = FALSE)
train <- data[randsample,]
valid <- data[-randsample,]
summary(train)
summary(valid)

# random forests
#install.packages("randomForest")
library(randomForest)

#names(data)[names(data) == "handicapped-infants"] <- "handicappedinfants"
names(train) <- gsub("-", "_", names(train))
train

model1 <- randomForest(class ~ ., data = train, importance = TRUE)
model1

pred <- predict(model1, train, type = "class")
table(pred, train$class)

names(valid) <- gsub("-", "_", names(valid))
pred <- predict(model1, valid, type = "class")
table(pred, valid$class)


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
pred3 <- predict(model1, test, type = "class")

result <- cbind(test$ID, as.character(pred3))
colnames(result) <- c("ID", "class")
write.table(result, file="res1.csv", col.names = TRUE, row.names = FALSE, sep = ",",quote=FALSE)