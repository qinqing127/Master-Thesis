library(tidyverse)
library(doParallel)
library(bit64)
library(ranger) #random forest package
library(sqldf)
library(data.table)
library(cvAUC)
library(foreach)
#read the data in 
load('development.RData')
data_s =dev %>% arrange(cvvar) %>% data.table()

##tuned parameter 
#specify minimum node sizes to seZrch over for tuning parameter selection
load('feat.RData')
load('minnode.RData')
load('ntry.RData')
cur.num.trees  = 100

sboot <- function(data, outcome){
  indi <- rep(0,nrow(data))
  ind1 <- data[get(outcome)==1, which=TRUE]
  ind0 <- data[get(outcome)==0, which=TRUE]
  booti <- data.table(ind=c(sample(ind1, replace=TRUE), sample(ind0, replace=TRUE)))
  booti <- booti[, .(freq=.N), by=ind] # make a frequency summary table
  indi[booti$ind] <- booti$freq
  return(indi) ## frequency of each observation to be chosen
}


## ----------------------step 3:model fitting with primary results-----------------------------
set.seed(79)
cv.list.f <- vector(mode = "list", length = cur.num.trees)
for(ii in 1:length(cv.list.f)){
  cv.list.f[[ii]] <- sboot(data_s, "death90")
}

set.seed(77)
rf_final <- ranger(dependent.variable.name="death90",  #specify outcome to predict
                   data = data_s[, c(feat,"death90"), with=FALSE],  #specify predctor data (subset all training data to exclude held-out fold and select only columns with predictors)
                   num.trees = cur.num.trees,      #specify number of trees
                   mtry= ntry,     #specify number of predictors to sample at each split (mtry)
                   importance="none",  #don't calculate variable importance measures
                   write.forest=TRUE,  #save the random forest object
                   probability=TRUE,  #calculate probabilities for terminal nodes (instead of a classificaiton tree that uses majority voting)
                   min.node.size= minnode,   #specify minimum node size
                   respect.unordered.factors="partition",   #for categorical variables, consider all grouping of categories (the alternative is that a categorical variable is turning into a number factor and treated like a continuous variable)
                   oob.error=FALSE,  
                   save.memory=FALSE,   # do not use memory saving options (can't recall why, but they don't work for some part of what we're doing here)
                   inbag=cv.list.f
)
rm(cv.list.f)
# predictions from testing set 
pre_final = predict(rf_final, data = data_s[,c(feat,'death90'), with = FALSE])$predictions[,2]
data_s$preds_final = pre_final
save(pre_final, file = 'preds_whole.RData')
save(rf_final, file = 'model0.RData')
rm(rf_final);gc()
##find the AUC on the testing set  
auc_final<-  AUC( pre_final,labels = data_s$death90) ## too high AUC here  check for random sample 

## find Sensitivity, Sepecificity, PPV and NPV 
#define percentiles that I am interested in: 99.9, 99.75, 99.5, 99
ps<-c(0.99, 0.95, 0.9, 0.75)

##functions to calculte sensitivity and others 
test_perf = function(ps, #threshold defined
                     outcome, ## testing dataset results 
                     pre_tr, ## predicted risk score for training set 
                     pre_te ## predicted risk score for the testing
){
  #define cutoffs based on the distribution of predictions
  cutoffs<-quantile(pre_tr, p=ps) ## only have one dataset 
  ns <- TP <- TN <- FP <- FN <- vector(length=length(cutoffs))
  for(j in 1:length(cutoffs)){
    ns[j] <- sum(pre_te >= cutoffs[j]) #number of observations with risk score above the threshold
    TP[j] <- sum(pre_te >= cutoffs[j] & outcome == 1) #number of true positives, risk score is above threshold and person had an event
    FP[j] <- sum(pre_te >= cutoffs[j] & outcome == 0) #number of false positives, risk score is above threshold but person did not have event
    FN[j] <- sum(pre_te < cutoffs[j] & outcome == 1) #number of false negatives, risk score is below threshold but person had event
    TN[j] <- sum(pre_te < cutoffs[j] & outcome == 0)  #number of true negatives, risk score is below threshold and person did not have an event
  }
  
  sensitivity = TP/(TP + FN); specificity = TN/(TN + FP)
  PPV = TP/(TP + FP) ;NPV = TN/(TN + FN)
  return(list(sensitivity = sensitivity, specificity = specificity, PPV = PPV, NPV = NPV))
}

all_per0 = test_perf(ps,data_s$death90,pre_final,pre_final)
#-- end of primary model for the whole data (Apparent performance)------------------------
print('End of primary model results')
#for 5 folds cv test performance
load('preds_5folds.RData')
test_perf_5folds = test_perf(ps,data_s$death90, pre_final,preds_best)

data_s$preds_5fold = preds_best


###-------------------Step 4: optimism calculation and 5-folds cv ci-------------------------
##prepare for the loop 
optimism_auc_boot = 1;optimism_other_boot = list()
auc_orignal = 0; all_per_original = list()
auc_samplem = 0;all_per_samplem = list()
auc_orignalbt = 0;all_per_originalbt = list()
## create element in the list 
optimism_other_boot2 = list(sensit99 = 1, sensit95 = 1, sensit90 = 1, sensit75 = 1,
                            specif99 = 1, specif95 = 1, specif90 = 1, specif75 = 1,
                            PPV99 = 1, PPV95 = 1, PPV90 = 1, PPV75 = 1,
                            NPV99 = 1, NPV95 = 1, NPV90 = 1, NPV75 = 1)
diffauc = 1;diff_other =rep(1,16)
# for the 5 cv methods parameter 
test_perf_cv = list(); auc_cv_best1 =0
#count the number in the loop
m=1
## when in a loop with 
set.seed(997)
while(m < 101 | diffauc > 0.0001 | length(which(diff_other > 0.0001) ) != 0){
  ##bootstrap the whole sample 
  sboot1 = sboot(data_s,'death90')
  sample1 = data.table(data_s[rep(1:nrow(data_s),sboot1)])
  
  ##calculate the modelo on the bootstrap data 
  pre_original = sample1$preds_final
  auc_orignal[m] = AUC(pre_original,labels = sample1$death90)
  all_per_original[[m]]= test_perf(ps,sample1$death90,pre_final,pre_original)
  
  ##predictions using the bootstrap sample
  cv.list1 <- vector(mode = "list", length = cur.num.trees)
  for(ii in 1:length(cv.list1)){
    cv.list1[[ii]] <- sboot(data_s, "death90")
  }
  
  rf_final1 <- ranger(dependent.variable.name="death90",  #specify outcome to predict
                      data = sample1[, c(feat,"death90"), with=FALSE],  #specify predctor data (subset all training data to exclude held-out fold and select only columns with predictors)
                      num.trees = cur.num.trees,      #specify number of trees
                      mtry= ntry,     #specify number of predictors to sample at each split (mtry)
                      importance="none",  #don't calculate variable importance measures
                      write.forest=TRUE,  #save the random forest object
                      probability=TRUE,  #calculate probabilities for terminal nodes (instead of a classificaiton tree that uses majority voting)
                      min.node.size= minnode,   #specify minimum node size
                      respect.unordered.factors="partition",   #for categorical variables, consider all grouping of categories (the alternative is that a categorical variable is turning into a number factor and treated like a continuous variable)
                      oob.error=FALSE,  
                      save.memory=FALSE,   # do not use memory saving options (can't recall why, but they don't work for some part of what we're doing here)
                      inbag=cv.list1
  )
  rm(cv.list1); gc()
  ##find the AUC on the bootstrap set 
  pre_final1 = predict(rf_final1, data = sample1[,c(feat,'death90'), with = FALSE])$predictions[,2]
  auc_final1<-  AUC( pre_final1,labels = sample1$death90)
  auc_samplem[m] = auc_final1
  ##find the AUC on the original data 
  pre_final1o = predict(rf_final1, data = data_s[,c(feat,'death90'), with = FALSE])$predictions[,2]
  auc_final1o<-  AUC( pre_final1o,labels = data_s$death90)
  auc_orignalbt[m] = auc_final1o
  ##calculate optimism for auc 
  optimism_auc = auc_final1 - auc_final1o
  
  ## calculate other test statisticcs 
  ## for the boostrap sample 
  test_p1 = test_perf(ps,sample1$death90,pre_final1,pre_final1) %>%
    unlist%>%matrix(nrow = 4,byrow = F)
  all_per_samplem[[m]] = test_p1
  ## for the original sample 
  test_p1o = test_perf(ps,data_s$death90,pre_final1,pre_final1o)%>%
    unlist%>%matrix(nrow = 4,byrow = F)
  all_per_originalbt[[m]] = test_p1o
  ## for the optimism calculation 
  optimism_other = test_p1 - test_p1o
  ##save the optimism value 
  optimism_other_boot[[m]] = optimism_other
  
  #calcaulte previous auc mean 
  auc_pre = mean(optimism_auc_boot)
  #save all AUC values in optimism_auc_boot
  optimism_auc_boot[m] = optimism_auc
  #calcaulte the difference of AUC
  diffauc = abs(mean(optimism_auc_boot) - auc_pre)
  
  # storing each test performance seperately
  mean_pre = sapply(optimism_other_boot2, mean)
  for(j in 1:16) {optimism_other_boot2[[j]][m] = optimism_other_boot[[m]][j]}
  ##calculate the difference of optimism for others 
  diff_other = abs(sapply(optimism_other_boot2, mean) - mean_pre)
  print(paste0('End of optimism ',m))
  rm(rf_final1) ;gc()
  
  
  #---------------for 5 folds cv 
  ##predictions using the bootstrap sample
  
  #calculate AUC and other test performance 
  preds_best1 = sample1$preds_5fold
  auc_cv_best1[m] = AUC(preds_best1, labels = sample1$death90)
  test_perf_cv[[m]] = test_perf(ps,sample1$death90,pre_final,preds_best1)
  rm(preds_best1)
  print(paste0('End of 5 folds cv ',m))
  m = m + 1
}


#-------Store all results as R data file 
#original model on the original data 
save(auc_final, file = 'AUC_original.RData'); save(all_per0, file = 'test.perf_original.RData')
#optimism 
save(optimism_auc_boot, file = 'optimism_auc_ci.RData')
save(optimism_other_boot2, file = 'test.perf_optimism_ci.RData' )
#original model on the boostrap data
save(all_per_original, file = 'test.perf_model0sample.RData')
save(auc_orignal,file ='AUC_model0sample.RData' )
#bootstrap model* on the original data 
save(all_per_originalbt, file = 'test.perf_modelsoriginal.RData')
save(auc_orignalbt, file = 'AUC_modelsoriginal.RData')
#bootstrap model* on the sample data* 
save(all_per_samplem, file = 'test.perf_modelssample.RData')
save(auc_samplem, file = 'AUC_modelssample.RData')
## 5 fold cv 
save(auc_cv_best1, file = 'AUC_cv_ci.RData'); save(test_perf_cv, file = 'test.perf_cv_ci.RData')
save(test_perf_5folds, file = 'test_perf_cv.Rdata')
