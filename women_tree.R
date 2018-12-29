# The dataset is Women's E-Commerce Clothing Reviews from Kaggle
# https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews/home

# The purpose of this code is to predict the recommendation of the product 
#   by inputing age, rating, division name, class name, department name

# Result
#   Precision: 0.9319307
#   Recall: 0.7492537
#   Accuracy: 0.9346391

# Explanation
#   As plot1 show, apparently, the reason why rating is the fist node is because it has highly corelation with the recommendation
#   Most of the people gave a recommendation when his/her rating is above 4
#   The interesting part is that younger people(below 50s) will not give a recommendation when his/her rating is 3
#   Some class of the product do have more probability to get a recommendation from the customer



# Start from importing the dataset
#   preprocess the data
#   setup the decision tree model
#   predict


rm(list = ls());gc();

###
library(dplyr)
library(data.table)
library(rpart)
library(rpart.plot)
library(rattle)
library(partykit)
library(dplyr)
library(quantmod)
library(caret)
library(xgboost)
library(Matrix)

# read csv
raw <- fread('/Users/AndrewHowCool/school/Big_Data/women/Womens_Clothing_E-Commerce_Reviews.csv')
raw <- as.data.frame(raw)
#View(raw)


### preprocess ###
# change the feature to factor
colnames(raw) <- c("v1", "clothing_id", "age", "title", "review_text", "rating", "recommended", "positive_feedback", "devision", "department", "class")
raw$recommended <- as.factor(raw$recommended) 
rating_order <- c("1","2","3","4","5")
raw$rating <- as.factor(ordered(raw$rating,level = rating_order))

# complete the dataset
df <- raw[complete.cases(raw), ]

# view df
names(df)
str(df)
summary(df)

# mutate a new age division
df <- mutate(df, age_div = case_when(
  age >= 18 & age < 30 ~ '18 ~ 29',
  age >= 30 & age < 40 ~ '30 ~ 39',
  age >= 40 & age < 50 ~ '40 ~ 49',
  age >= 50 & age < 60 ~ '50 ~ 59',
  age >= 60 & age < 70 ~ '60 ~ 69',
  age >= 70 & age < 80 ~ '70 ~ 79',
  age >= 80 & age < 90 ~ '80 ~ 89',
  age >= 90 & age < 1000 ~ '90 ~ 99',
  is.na(age) ~ 'Unknown'
))

# select age_div, recommended ind, rating, division name, class name, department name

df2 <- df
df <- select(df, age_div, recommended, devision, class, department, rating)
#View(df2)
#View(df)






### setup the decision tree model ###
#  train = 0.8, test = 0.2 
set.seed(22)
train.index <- sample(x=1:nrow(df), size=ceiling(0.8*nrow(df) ))
train <- df[train.index, ]
test <- df[-train.index, ]

# cart model
cart.model<- rpart(formula = rating ~. , 
                   control = rpart.control(cp = 0.001),
                   data=train)

# output each node's detail 
cart.model

# plot 1
fancyRpartPlot(cart.model)

# plot 2
rparty.tree <- as.party(cart.model) # transform the cart model
rparty.tree # output each node's detail
plot(rparty.tree) 






### predict ###
result <- predict(cart.model, newdata = test, type = "class")
# confusion matrix
cm <- table(test$rating, result, dnn = c("real", "predict"))
cm

# Precision
precision1 <- cm[1, 1] / sum(cm[, 1])
precision2 <- cm[2, 2] / sum(cm[, 2])
precision3 <- cm[3, 3] / sum(cm[, 3])
precision4 <- cm[4, 4] / sum(cm[, 4])
precision5 <- cm[5, 5] / sum(cm[, 5])

# Recall
recall1 <- cm[1, 1] / sum(cm[1, ])
recall2 <- cm[2, 2] / sum(cm[2, ])
recall3 <- cm[3, 3] / sum(cm[3, ])
recall4 <- cm[4, 4] / sum(cm[4, ])
recall5 <- cm[5, 5] / sum(cm[5, ])

# Accuracy
accuracy <- sum(diag(cm)) / sum(cm)




### sentiment analysis ###
library(tidytext)
library(tidyverse)
library(glue)
library(stringr)


df2 <- data.frame(lapply(df2, as.character), stringsAsFactors=FALSE)


df2 <- df2 %>%
  unnest_tokens(word, review_text) %>%
  anti_join(stop_words) %>%
  inner_join(get_sentiments("bing")) %>%
  mutate(sent = if_else(sentiment == "positive", 1, -1)) %>%
  group_by(v1, clothing_id, age, age_div, recommended, positive_feedback, devision, class, department, rating) %>%
  summarise(sum_sent = sum(sent))
#View(df2)

str(df2)

# test of independence between rating and sum_sent
itest <- data_frame(class = as.factor(df2$class), rating = df2$rating, sum_sent = df2$sum_sent)
kk <- filter(itest, rating == 1 | rating == 2 | rating == 3 | rating == 4)
kk <- filter(kk, sum_sent<7)
kk$sum_sent <- as.factor(kk$sum_sent)
kk$rating <- as.factor(kk$rating)
table(select(kk, rating, sum_sent))
chisq.test(table(select(kk, rating, sum_sent))) # p-value < 2.2e-16 => sum_sent and rating are independent

# plot mrating and msum_sent
library(ggplot2)
mean_itest <- group_by(kk, class) %>%
  summarise(mrating = mean(as.numeric(rating)), msum_sent = mean(as.numeric(sum_sent))) %>%
  as.data.frame()

p <- ggplot(mean_itest, aes(x = class, group = 1)) +
  geom_line(aes(y = (msum_sent-mean(msum_sent))/sd(msum_sent), color = "sentiment")) +
  geom_line(aes(y = (mrating-mean(mrating))/sd(mrating), color = "rating")) 
p






# testing playground
# testing playground

rating_order <- c("1","2","3")

train.index <- sample(x=1:nrow(df2), size=ceiling(0.8*nrow(df) ))

test <- df2[-train.index, ]

#?Nrating???Ƥ��??T?Ӱ϶?
train_df <- df2[train.index, ] %>%
  group_by() %>%
  mutate(age = as.numeric(age),
         positive_feedback = as.numeric(positive_feedback),
         #rating_recom = paste0(rating,"_",recommended),
         rating_1st = if_else(as.numeric(rating) >= 4,3,if_else(as.numeric(rating) == 3,2,1)),
         #rating = as.factor(as.numeric(rating)/5),
         rating = as.factor(ordered(rating_1st,level = rating_order))) %>%
  select(-c(v1,clothing_id,rating_1st)) %>% na.omit

str(train_df)
x_train <- train_df %>% select(-rating) %>% as.matrix()
y_train <- train_df$rating %>% as.matrix()
trainData_first <- list(data = as(x_train, "dgCMatrix"),label = y_train)
trainData <- xgb.DMatrix(trainData_first$data,label = trainData_first$label)




#?Nrating???Ƥ��??T?Ӱ϶?
test_df <- df2[-train.index, ] %>%
  group_by() %>%
  mutate(age = as.numeric(age),
         positive_feedback = as.numeric(positive_feedback),
         #rating_recom = paste0(rating,"_",recommended),
         rating_1st = if_else(as.numeric(rating) >= 4,3,if_else(as.numeric(rating) == 3,2,1)),
         #rating = as.factor(as.numeric(rating)/5),
         rating = as.factor(ordered(rating_1st,level = rating_order))) %>%
  select(-c(v1,clothing_id,rating_1st)) %>% na.omit

str(test_df)
x_test <- test_df %>% select(-rating) %>% as.matrix()
y_test <- test_df$rating %>% as.matrix()
testData <- list(data = as(x_test, "dgCMatrix"),label = y_test)



paramTable <- expand.grid(eta = c(0.02,0.06,0.1), #0.06
                          max_depth = c(2,5,10),  #2     
                          subsample = c(0.6), 
                          colsample_bytree = c(0.6))
##0.02, 2, 0.6, 0.6

# ?i?????e???ҬD???̨ΰѼ?
cvOutput <- NULL
for(iy in c(1:nrow(paramTable))){
  
  
  
  #?Ѽƪ?
  params <- list(booster = "gbtree",
                 eta = paramTable$eta[iy], 
                 max_depth = paramTable$max_depth[iy], 
                 subsample = paramTable$subsample[iy], 
                 colsample_bytree = paramTable$colsample_bytree[iy], 
                 "eval_metric" = "mlogloss",
                 objective = "multi:softmax",
                 "num_class" = 4)
  
  #Cross validation
  cvResult <- xgb.cv(params = params, 
                     data = trainData, 
                     nrounds = 1000, 
                     nfold = 5, 
                     early_stopping_rounds = 10, 
                     verbose = 1)
  
  #CV???G
  cvOutput <- cvOutput %>%
    bind_rows(tibble(paramsNum = iy,
                     bestIteration = cvResult$best_iteration,
                     bestCvlog = cvResult$evaluation_log$train_mlogloss_mean[bestIteration],
                     bestCvmae_test = cvResult$evaluation_log$test_mlogloss_mean[bestIteration],
                     eta = paramTable$eta[iy], 
                     max_depth = paramTable$max_depth[iy], 
                     subsample = paramTable$subsample[iy], 
                     colsample_bytree = paramTable$colsample_bytree[iy]))
  print(tail(cvOutput,10))
}

xx<-cvOutput %>% arrange(bestCvmae_test)

# ---------------------------------------------------???e???ҳ̨ΰѼ?-------------------------------------------------------------------
bestCvSite <- which(cvOutput$bestCvmae_test == min(cvOutput$bestCvmae_test))
bestCvmae <- cvOutput$bestCvlog[bestCvSite]
bestIteration <- cvOutput$bestIteration[bestCvSite]
bestParamsNum <- cvOutput$paramsNum[bestCvSite]

#bestCvSite <- which(cvOutput$bestCvmae_test == min(cvOutput$bestCvmae_test))
#bestCvmae <- cvOutput$bestCvmae[bestCvSite]
#bestIteration <- cvOutput$bestIteration[bestCvSite]
#bestParamsNum <- cvOutput$paramsNum[bestCvSite]
# ---------------------------------------------------???e???ҳ̨ΰѼ?-------------------------------------------------------------------
# ?̨ΰѼƲզX
my_params <- list(booster = "gbtree", 
                  eta = paramTable$eta[bestParamsNum],
                  #eta = 0.06,
                  max_depth = paramTable$max_depth[bestParamsNum],
                  #max_depth = 1,
                  subsample = paramTable$subsample[bestParamsNum], 
                  #subsample = 0.6,
                  colsample_bytree = paramTable$colsample_bytree[bestParamsNum], 
                  #colsample_bytree = 0.6,
                  #"eval_metric" = "mae",
                  #objective = "reg:linear",
                  "eval_metric" = "mlogloss",
                  objective = "multi:softmax",
                  "num_class" = 4)

# xgboost?ҫ??V?m
xgbModel_valid <- xgb.train(data = trainData,
                            params = my_params,
                            maximize = FALSE,
                            #nrounds = 200,
                            nrounds = bestIteration)

importance_matrix <- xgb.importance(colnames(trainData_first$data),model = xgbModel_valid)

print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix[1:20,])


#-----------------------------------------------------validation -----------------------------------------------------------------------
#?w????
test_rating = (predict(xgbModel_valid, testData$data)) %>% as.numeric()

#?w??????
test_table <- tibble(test_ret = test_rating) %>% mutate(act_ret = testData$label) %>%
  mutate(true = as.numeric(test_ret == act_ret))
cm2 <- table(test_table$act_ret,test_table$test_ret, dnn = c("real", "predict"))

# Precision
pc1 <- cm2[1,1] / sum(cm2[, 1])
pc2 <- cm2[2,2] / sum(cm2[, 2])
pc3 <- cm2[3,3] / sum(cm2[, 3])
pc4 <- cm2[4,4] / sum(cm2[, 4])
pc5 <- cm2[5,5] / sum(cm2[, 5])

# Recall
rc1 <- cm2[1,1] / sum(cm2[1, ])
rc2 <- cm2[2,2] / sum(cm2[2, ])
rc3 <- cm2[3,3] / sum(cm2[3, ])
rc4 <- cm2[4,4] / sum(cm2[4, ])
rc5 <- cm2[5,5] / sum(cm2[5, ])

ACC <- mean(test_table$true) # 0.6154529

### compare f1 score ###
# rating 1
2*precision1*recall1/(precision1+recall1)
2*pc1*rc1/(pc1+rc1)

# rating 2
2*precision2*recall2/(precision2+recall2)
2*pc2*rc2/(pc2+rc2)

# rating 3
2*precision3*recall3/(precision3+recall3)
2*pc3*rc3/(pc3+rc3)

# rating 4
2*precision4*recall4/(precision4+recall4)
2*pc4*rc4/(pc4+rc4)

# rating 5
2*precision5*recall5/(precision5+recall5)
2*pc5*rc5/(pc5+rc5)


#  0.8639456   factor 0.7639752
ACC
