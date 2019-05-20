#install.packages("magrittr") 
#install.packages("dplyr")    # alternative installation of the %>%
#install.packages("reprex")
#install.packages("rminer")
#install.packages("rpart.plot")
library(magrittr) # need to run every time you start R and want to use %>%
library(dplyr)
library(purrr)
library(ggplot2)
library(gridExtra)
library(reprex)
library(rminer)
library(e1071)
library(rpart)
library(rattle)
library(rpart.plot)
library(tibble)
library(party)
library(dplyr)
library(ggraph)
library(igraph)



# Upload data
##########################################################################
##########################################################################
# Load train data from My PC;
path_train <- "./amazon_review_ID.shuf.lrn.csv"
amazon_train_csv <- read.csv(path_train, stringsAsFactors = FALSE)
str(amazon_train_csv) # none of the columns are labeled
class(amazon_train_csv)
sum(is.na(amazon_train_csv))
apply(is.na(amazon_train_csv), 2, which)
summary(amazon_train_csv$ID)
summary(amazon_train_csv$V1)
summary(amazon_train_csv$Class)
print(amazon_train_csv$Class)
print(sort(amazon_train_csv$Class))
unique(amazon_train_csv$Class)
# Convert data.frame column format from character to factor
amazon_train_csv[,'Class'] <- as.factor(amazon_train_csv[,'Class'])
str(amazon_train_csv$Class)
unique(amazon_train_csv$Class)
# print(sort(amazon_noId$Class))
levels(amazon_train_csv$Class)
amazon_train_class <- as.numeric(amazon_train_csv$Class)
hist(amazon_train_class)


##########################################################################
# Load test data from My PC;
path_test <- "amazon_review_ID.shuf.tes.csv"
amazon_test_csv <- read.csv(path_test, stringsAsFactors = FALSE)
str(amazon_test_csv) # none of the columns are labeled
class(amazon_test_csv)
sum(is.na(amazon_test_csv))
apply(is.na(amazon_test_csv), 2, which)
summary(amazon_test_csv$ID)
summary(amazon_test_csv$V1)
# Obs: where you need to predict class
summary(amazon_test_csv$Class)
print(amazon_test_csv$Class)
print(sort(amazon_test_csv$Class))
unique(amazon_test_csv$Class)

##########################################################################
##########################################################################
library(randomForest)
library(caret)

#Random Forest Model with all features
load("amazonRF_full.RData")
amazon_RF_full <- randomForest(Class ~ ., data = amazon_train_csv, ntree=100, do.trace=100, importance=TRUE, proximity=TRUE)
save(amazon_RF_full, file = "amazonRF_full.RData")
plot(amazon_RF_full)
getTree(amazon_RF_full, 1, labelVar=TRUE)
attributes(amazon_RF_full)
# Variable Importance
varImpPlot(amazon_RF_full, sort = TRUE, n.var = 200, main = "The 200 variables with the most predictive power")
importance(amazon_RF_full)
importance    <- importance(amazon_RF_full)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))
#Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
str(rankImportance)
summary(rankImportance$Importance)
hist(rankImportance$Importance)
sort(rankImportance$Importance, decreasing = TRUE)
rankImportance_subset <- subset(rankImportance, Importance >= 0.1)
str(rankImportance_subset)
sapply(rankImportance_subset, class)
sapply(rankImportance_subset, typeof)
important_features <- as.character(rankImportance_subset$Variables)
str(important_features)
print(important_features)
#Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 
# To choose Best Features, we used rfcv() function of package randomForest.
#load("featuresSelection.RData")
#fo <- rfcv(amazon_train_csv[,-10001],amazon_train_csv[,10001], cv.fold=10, scale="log", step=0.9)
#best <- which.max(fo$error.cv)
#save(best, file = "featuresSelection.RData")
#plot( fo$n.var,rfo$error.cv, type = "h", main = "importance", xlab="number of features", ylab = "classifier error rate")
#axis(1, best, paste("best", best, sep="\n"), col = "red", col.axis = "red")

# Make prediction
pred_train <- predict(amazon_RF_full, amazon_train_csv, type = "Class")
str(pred_train)
summary(pred_train)
str(amazon_train_csv$Class)
summary(amazon_train_csv$Class)
# Create Confusion Matrix for train dataset
table(pred_train, amazon_train_csv$Class)

pred_test <- predict(amazon_RF_full, amazon_test_csv, type = "Class")
str(pred_test)
summary(pred_test)
summary(amazon_test_csv$Class)
# Create Confusion Matrix
table(pred_test, amazon_test_csv$Class)

# Save the results
result <- cbind(amazon_test_csv$ID, as.character(pred_test))
colnames(result) <- c("ID", "class")
write.table(result, file="res1.csv", col.names = TRUE, row.names = FALSE, sep = ",",quote=FALSE)




# The bellow code was not run
# Random Forest Model with feature selection
load("amazon_RF_featureSelection.RData")
# Obs: variable lengths differ (found for 'important_features')
# important_features is the vector containing top feature to be included in bellow model
amazon_RF_featureSelection <- randomForest(Class ~ important_features, data = amazon_train_noId, ntree=1000, do.trace=100, proximity=TRUE)
save(amazon_RF_featureSelection, file = "amazon_RF_featureSelection.RData")
plot(amazon_RF_featureSelection)
getTree(amazon_RF_featureSelection, 1, labelVar=TRUE)
attributes(amazon_RF_featureSelection)







sessionInfo()


### attila



set.seed(123)
randsample <- sample(nrow(amazon_train_csv), 0.8*nrow(amazon_train_csv), replace = FALSE)
train <- amazon_train_csv[randsample,]
valid <- amazon_train_csv[-randsample,]

#baseline

baseline <- sample(train$Class, nrow(valid), replace = TRUE)
cm <- confusionMatrix(as.factor(baseline), valid$Class)
cm$overall
  
#rf
usefeatures <- important_features[2:1004]
formula <- paste("Class ~", paste(usefeatures, collapse = '+'))
start_time <- Sys.time()
model_rf_important <- randomForest(as.formula(formula), data = train, ntree=100, do.trace=100, importance=TRUE, proximity=TRUE)
end_time <- Sys.time()
end_time - start_time
#load("amazonRF_full.RData")
cm <- confusionMatrix(predict(model_rf_full, valid, type = "class"), valid$Class)
cm$overall

#nb

start_time <- Sys.time()
model_nb_full <- naiveBayes(Class ~ . - ID, data=train)
end_time <- Sys.time()
end_time - start_time

cm <- confusionMatrix(predict(model_nb_full, valid, type = "class"), valid$Class)
cm$overall

#nb with first x features
accs <- list()
for (n in seq(100, 2500, by=100)) {
  print(n)
  usefeatures <- important_features[2:n]
  formula <- paste("Class ~", paste(usefeatures, collapse = '+'))
#  print(formula)
  model_nb_important <- naiveBayes(formula=as.formula(formula), data=train)
  cm <- confusionMatrix(predict(model_nb_important, valid, type = "class"), valid$Class)
  print(cm$overall)
  accs <- c(accs, cm$overall[1])
}
op <- par(mar=c(4,4,4,4)) 
plot(seq(100, 2500, by=100), accs, ylab = "accuracy", xlab = "number of features")
plot(seq(100, 2500, by=100), accs, col=ifelse(accs>0.5, "red", "black"), ylab = "accuracy", xlab = "number of features")
abline(v=1300, col="red", lty=3)
mtext("1300", side=1, line=1, col="red")
rm(op)

start_time <- Sys.time()
usefeatures <- important_features[2:1300]
formula <- paste("Class ~", paste(usefeatures, collapse = '+'))
model_nb_900 <- naiveBayes(formula=as.formula(formula), data=train)
end_time <- Sys.time()
end_time - start_time
cm <- confusionMatrix(predict(model_nb_900, valid, type = "class"), valid$Class)
print(cm$overall)


#knn

#library(class)

#pred_knn <- knn(train[,2:17], valid_numeric[,2:17], train_numeric$class,k=k)

#accs <- list()
#for (k in 2:16) {
#  pred_knn <- knn(train_numeric[,2:17], valid_numeric[,2:17], train_numeric$class,k=k)
#  cm <- confusionMatrix(pred_knn, valid_numeric$class)
#  accs <- c(accs, cm$overall[1])
#}


