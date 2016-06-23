---
title: "Practical Machine Learning Course Project"
categories:
  - Markdown
---

## Data Processing

First we download the data from Coursera and change the directory to where you place the data files. Then we read both training and testing data sets.


```r
library(caret)
library(randomForest)
set.seed(2323)
training = read.csv("pml-training.csv", header = TRUE, na.strings = c('', NA, '#DIV/0!'));
testing = read.csv("pml-testing.csv", header = TRUE, na.strings = c('', NA, '#DIV/0!'));
```

In order to do the analysis and set up the prediction model, we need to exclude unnecessary columns and rows. First, to make fully use of all observations, we only use variables that do not contain any NA in them. Second, if we take a look at the test data set, we notice that all the observations in the test data set have value 'no' for the ¡°new_window¡± variable. Thus, it is reasonale for us to only pick rows that has a 'no' value for the ¡°new_window¡± variable in the training data set. After this we remove some obviously irrelevant variables. Finally, we look at the correlations and remove highly correlated columns.


```r
training = training[,colSums(is.na(training))==0];
training = training[training$new_window == "no",];

# Take a look at the variables left and remove irrelevant variables.
names(training);
training = training[, !(colnames(training) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window"))];

# Remove highly correlated columns.
library(caret)

correlated = findCorrelation(cor(training[, !(colnames(training) %in% c("classe"))]), cutoff = 0.75);
training = training[, -correlated];

# Turn variable classe into factors.
training$classe = factor(training$classe);
```

Now we have our data set ready for building the model.

## Exploratory Analysis and Model Selection

Below are somw exploratory graphs for the data set and we can easily notice a clustering feature. So it is reasonable to choose random forest to build the model.


```r
library(gridExtra)

grid.arrange(
p1 = ggplot(data = training, aes(x = accel_forearm_z, y = gyros_arm_x, colour = classe)) + geom_point(),
p2 = ggplot(data = training, aes(x = pitch_arm, y = magnet_forearm_z, colour = classe)) + geom_point(),
p3 = ggplot(data = training, aes(x = total_accel_forearm, y = roll_dumbbell, colour = classe)) + geom_point(),
p4 = ggplot(data = training, aes(x = magnet_dumbbell_z, y = yaw_forearm, colour = classe)) + geom_point(),
ncol = 2
)
```

![plot of chunk exploratory](figure/exploratory.png) 

In order to estimate the out-sample error of the model, we use 8-fold cross validation. 


```r
fold_list = createFolds(training$classe, k = 8, list=TRUE, returnTrain=FALSE);
modelResult = list();

for(i in 1:8){
  fold = fold_list[[i]];
  modFit = randomForest(classe~., data = training[-fold,]);
  modPre = predict(modFit, training[fold,]);
  result = confusionMatrix(modPre, training[fold, "classe"]);
  modelResult[[i]] = list(
    modelFit = modFit,
    modelPredict = modPre,
    model_result = result);
}
```

After the model, we take a look at the accuracy and error rate. 

```r
mean_accuracy = mean(sapply(modelResult, function(x){return(x$model_result$overall[[1]])}));
sd = sd(sapply(modelResult, function(x){return(x$model_result$overall[[1]])}));

lower_bound = mean_accuracy - sd/sqrt(8)*1.96;
upper_bound = mean_accuracy + sd/sqrt(8)*1.96;
interval = c(lower_bound, upper_bound);
interval
```

```
## [1] 0.9942 0.9954
```

So with 95% confidence, we can say the accuracy rate of the model is within (0.9942, 0.9954).

## Prediction

Now we use the model we get to do the prediction on testing data set. 


```r
models = sapply(modelResult, function(x){return(x$model_result$overall[[1]])});
best = which(models==max(models))[1];
bestModel = modelResult[[best]]$modelFit;

prediction = predict(bestModel, testing);
```

