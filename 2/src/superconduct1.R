library(readr)
url <- "2/data/superconduct/train.csv"

data <- read_csv(url)
str(data)
summary(data)

# linear regression with all variables

m1 <- lm(critical_temp~.,data=data)
summary(m1)
plot(m1)

plot(data$critical_temp, predict(m1), xlab="y", ylab="y-hat")
abline(c(0,1), col="red")

# use stepwise feature selection based on AIC
m1s <- step(m1)
summary(m1s)

# dropped variables
m1s$anova

plot(data$critical_temp, predict(m1s), xlab="y", ylab="y-hat")
abline(c(0,1), col="red")


#robust regression

library(robustbase)

# lts
m2 <- ltsReg(critical_temp~.,data=data)

summary(m2)
plot(m2)

# lmrob

m3 <- lmrob(critical_temp~.,data=data)

summary(m3)
plot(m3)


# cv validation

library(caret)
train_control <- trainControl(method="cv", number=10)
model <- train(critical_temp~., data=data, trControl=train_control, method = 'lm')
print(model)
plot(model)