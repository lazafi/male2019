library(readxl)
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
m_flw <- lm(Followers~., data=data_nona[,2:14])
### eval predictions
plot(data_nona$Followers, predict(m_flw))
abline(c(0,1), col="red")
#### not good, use median instead to fill in missing values
data[which(is.na(data$Followers)),14] <- median(data$Followers, na.rm = TRUE)


## Screens
m_sc1 <- lm(Screens~., data=data_nona[,2:14])
m_sc1

### eval predictions
plot(data_nona$Screens, predict(m_sc1))
abline(c(0,1), col="red")
#### good enought 
filldata_screens <- data[which(is.na(data$Screens)),]
data[which(is.na(data$Screens)),7] <- predict(m_sc1, newdata=filldata_screens[,2:14])


##Budget
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

model <- train(formula, data=data, trControl=train_control, method = 'lm')
model$results
plot(data$Gross, predict(model$finalModel))
abline(c(0,1), col="red")

# try with centered and scaled values
model2 <- train(formula, data=data, trControl=train_control, method = 'lm', preProcess = c("center", "scale"))
model2$results
plot(data$Gross, predict(model2$finalModel))
abline(c(0,1), col="red")

# robust regression
model_rob <- train(formula, data=data, trControl=train_control, method = 'rlm',  preProcess = c("center", "scale"))
model_rob$results

plot(data$Gross, predict(model_rob$finalModel), pch = model_rob$finalModel$weights)
abline(c(0,1), col="red")

# random forrest

grid_rf <- expand.grid(
  mtry = c(1, 10, 20, 50, 100)
)

model_rf <- train(
  formula,
  data=data,
  trControl = train_control,
  tuneGrid = grid_rf,
  verbose = TRUE
)

plot(model_rf$results$mtry, model_rf$results$Rsquared, xlab="mtry", ylab="r²", log = "x", type = "b")

plot(data$Gross, predict(model_rf$finalModel), xlab="y", ylab="y-hat")
abline(c(0,1), col="red")

#######
