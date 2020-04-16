library(tidyverse)
library(doParallel)
library(bit64)
library(ranger) #random forest package
library(sqldf)
library(data.table)
library(cvAUC)
library(foreach)

##----------------------------STEP 1-------------------------------------------
load('development.RData')
data_s = dev %>% arrange(cvvar) 
##training and testing sets seperation seperation using the already build in folds
training = data_s[data_s$training == 1, ] %>% data.table()
testing = data_s[data_s$training == 0, ] %>% data.table() v


##----------------------------STEP 2---------------------------------
#Tuning the parameter minimal node size in the traning set 
feat <- setdiff(colnames(training),c("person_id","visit_seq","death30", "death90", "training", "cvvar", "everdied"))

#specify number of trees for random forest. I expect 100 will be enough
cur.num.trees <- 100 

#specify minimum node sizes to seZrch over for tuning parameter selection
min.node.size <- c(10000, 25000, 50000,100000,150000) ##change this for whole dataset 

#specify number of features to randomly sample at each split of tree
#for this dataset, the default number of predictors that random forest would select is sqrt(p)=11 (where p is number of predictors)
#Eric's code searches over 3*sqrt(p), 2*sqrt(p), sqrt(p), and sqrt(p)/2 where the floor() functin rounds each to the lower whole number
node.vars <- c(floor(sqrt(length(feat)))*2, floor(sqrt(length(feat))), floor(sqrt(length(feat))/2))

#function from Rod that performs a stratfied bootstrap and saves information so that it can be used to tell ranger bootstrap samples each tree should be built on
#you will use this function for 5-fold CV to select tuning parameters and on the entire training set, so "data" here is not the full training dataset. For CV, you will subset to the 4 folds you are training on, excluding visits from the 5th 
sboot <- function(data, outcome){
  indi <- rep(0,nrow(data))
  ind1 <- data[get(outcome)==1, which=TRUE]
  ind0 <- data[get(outcome)==0, which=TRUE]
  booti <- data.table(ind=c(sample(ind1, replace=TRUE), sample(ind0, replace=TRUE)))
  booti <- booti[, .(freq=.N), by=ind] # make a frequency summary table
  indi[booti$ind] <- booti$freq
  return(indi) ## frequency of each observation to be chosen
}

#cv.list is going to be a list containing 5 lists (one for each CV fold)
#For each CV fold, the list will contain 100 elements (100=cur.num.trees) that specify the bootstrap sample for each tree to ranger
#For each tree, the element is a vector with length = number of visits in the dataset (nrow(data) in line 30) with tone fold removed
#these vector contain indi from the sboot function, that is, indicates how many times each visit should appear in the bootstrap sample
set.seed(97)
cv.list <- foreach(i=1:5) %do% {
  samp.list <- vector(mode = "list", length = cur.num.trees)
  for(ii in 1:length(samp.list)){
    print(ii)
    samp.list[[ii]] <- sboot(training[cvvar != i,], "death90")
  }
  samp.list
}

#this code will do 5-fold CV, it estimates random forests for each commbination of tuning parameters and saves predictions for the left-out fold
set.seed(77)
auc_cv <- foreach(j=min.node.size, .combine=cbind) %do% {   #code cycles through all options for min.node.size
  auc_nv <- foreach(i=node.vars, .combine=cbind) %do% {    #code cycles through all options for all node.vars
    preds <- foreach(k=1:5, .combine=c) %do% {  #code cylces through CV indices 1-5
      #this next function estimates a random forest model with the selected min.node.size, node.vars. and excluded CV fold
      rf_mod <- ranger(dependent.variable.name="death90",  #specify outcome to predict
                       data = training[cvvar != k, c(feat,"death90"), with=FALSE],  #specify predctor data (subset all training data to exclude held-out fold and select only columns with predictors)
                       num.trees = cur.num.trees,      #specify number of trees
                       mtry=i,     #specify number of predictors to sample at each split (mtry)
                       importance="none",  #don't calculate variable importance measures
                       write.forest=TRUE,  #save the random forest object
                       probability=TRUE,  #calculate probabilities for terminal nodes (instead of a classificaiton tree that uses majority voting)
                       min.node.size=j,   #specify minimum node size
                       respect.unordered.factors="partition",   #for categorical variables, consider all grouping of categories (the alternative is that a categorical variable is turning into a number factor and treated like a continuous variable)
                       oob.error=FALSE,   #do not calculate out of bag statistics
                       save.memory=FALSE,   # do not use memory saving options (can't recall why, but they don't work for some part of what we're doing here)
                       inbag=cv.list[[k]]) #specify the inbag sample for this each of 100 trees =  the stratified bootstrap samples we got from sboot()
      print(c(i,j, k)) #trace for progress
      pre_cv = predict(rf_mod,training[cvvar == k, c(feat), with=FALSE]) #make predictions for the held-out fold from rf_mod fit above
      pre_cv$predictions[,2]  
       } #end preds
    AUC (preds, labels = training %>% arrange(cvvar) %>% select(death90) )
  } #end auc_nv 
  auc_nv
} #end outer foreach

# find the biggest the auc value
which(auc_cv == max(auc_cv)) 
save(auc_cv, file = 'Tunedauc_split.RData')
#once n trees and mtry, minimal node size is decided from the auc.cv 
#THere are in total 12 combination of AUCs 
#(3*sqrt(p), 500), (2*sqrt(p), 500), (sqrt(p), 500), (sqrt(p)/2, 500)
#(3*sqrt(p), 1000), (2*sqrt(p), 1000), (sqrt(p), 1000), (sqrt(p)/2, 1000))
#(3*sqrt(p), 2500), (2*sqrt(p), 2500), (sqrt(p), 2500), (sqrt(p)/2, 2500)
 l = length(node.vars)
if(which(auc_cv == max(auc_cv)) %% l == 0){
  ntry = node.vars[l]
}else{
  ntry = node.vars[which(auc_cv == max(auc_cv)) %% l]
} # if the reminder is 0, choose the last one in all
minnode = min.node.size[ceiling(which(auc_cv == max(auc_cv)) / l)]
save(minnode, file = 'minnode_split.RData'); save(ntry, file = 'ntry_split.RData')
##------------------------finish tunning the parameter ---------------------------


## ---------------------- STEP 3 ---------------------------------------------------
#Fit the model on the whole traning set, 
set.seed(79)
cv.list.f <- vector(mode = "list", length = cur.num.trees)
  for(ii in 1:length(cv.list.f)){
    print(ii)
    cv.list.f[[ii]] <- sboot(training, "death90")
  }


set.seed(79)
rf_final <- ranger(dependent.variable.name="death90",  #specify outcome to predict
                 data = training[, c(feat,"death90"), with=FALSE],  #specify predctor data (subset all training data to exclude held-out fold and select only columns with predictors)
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
# predictions from testing set 
pre_final = predict(rf_final, data = testing[,c(feat,'death90'), with = FALSE])$predictions[,2]
##find the AUC on the testing set  
auc_final<-  AUC( pre_final,labels = testing$death90)
save(rf_final, file = 'model_split.RData')
## calculate the threshold based Sensitivity, Specifity, PPV and NPV 
#-------------calculate sensitivity, specificity, PPV and NPV 

#define percentiles that I am interested in: 99.9, 99.75, 99.5, 99
ps<-c(0.99, 0.95, 0.9, 0.75)

#define cutoffs based on the distribution of predictions
pre_train = predict(rf_final, data = training[,c(feat,'death90'), with = FALSE])$predictions[,2]
save(pre_train, file = 'preds_split.RData')
#in sample auc 
auc_insample = AUC(pre_train, labels = training$death90)
save(auc_insample, file='auc_insample_split.RData')

cutoffs<-quantile(pre_train, p=ps)
##functions to calculte sensitivity and others 
test_perf = function(cutoff, #cutoffs defined in the training 
                     outcome, ## testing dataset results 
                     pre_te ## predicted risk score for the testing
){
  ns <- TP <- TN <- FP <- FN <- vector(length=length(cutoff))
  for(j in 1:length(cutoff)){
    ns[j] <- sum(pre_te >= cutoff[j]) #number of observations with risk score above the threshold
    TP[j] <- sum(pre_te >= cutoff[j] & outcome == 1) #number of true positives, risk score is above threshold and person had an event
    FP[j] <- sum(pre_te >= cutoff[j] & outcome == 0) #number of false positives, risk score is above threshold but person did not have event
    FN[j] <- sum(pre_te < cutoff[j] & outcome == 1) #number of false negatives, risk score is below threshold but person had event
    TN[j] <- sum(pre_te < cutoff[j] & outcome == 0)  #number of true negatives, risk score is below threshold and person did not have an event
  }
  
  sensitivity = TP/(TP + FN); specificity = TN/(TN + FP)
  PPV = TP/(TP + FP) ;NPV = TN/(TN + FN)
 return(list(sensitivity = sensitivity, specificity = specificity, PPV = PPV, NPV = NPV))
}

all_per0 = test_perf(cutoffs,testing$death90,pre_final )

##----------------sTEP 4-----------------------------------
#bootstrap the whole testing dateset and calculate the CI 

testing1 = cbind(testing$death90,pre_final) %>% data.table()
colnames(testing1) = c('death90','pre_final')
set.seed(97)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
boot_ci <- foreach(b=1:250, .verbose=T) %dopar% {
   library(data.table)
   ##bootstrap sampling
   boot1 = sboot(testing1,'death90')
   testingb = data.table(testing1[rep(1:nrow(testing1),boot1)])
   ##find the AUC on the testing set
   library(cvAUC)
   auc_final_c<-  AUC(testingb$pre_final,labels = testingb$death90)

  ##calculate sensitivity, specificity, PPV and NPV 
  test_other =  test_perf(cutoffs,testingb$death90,testingb$pre_final)
  test_other['auc_final'] = auc_final_c
  test_other
  
}
stopCluster(cl)

#create empty matrix and vector to save information
auc_final_c= rep(0,250)
sensitivity_c = matrix(nrow = 250,ncol = 4);specificity_c = matrix(nrow = 250,ncol = 4)
PPV_c = matrix(nrow = 250,ncol = 4) ;NPV_c = matrix(nrow = 250,ncol = 4)
#saving all paramters into corresponding listings 
for(b in 1:250){
  auc_final_c[b] <- boot_ci[[b]]$auc_final
  sensitivity_c[b, ] <- boot_ci[[b]]$sensitivity
  specificity_c[b, ] <- boot_ci[[b]]$specificity
  PPV_c[b, ] <- boot_ci[[b]]$PPV
  NPV_c[b, ] <- boot_ci[[b]]$NPV
}

#make all test performance into a list(except aucaic)
all_per = list(sensitivity_c = sensitivity_c, specificity_c = specificity_c,
         PPV_c = PPV_c,NPV_c = NPV_c)

#function to get 95% ci for each parameter 
quar = function(par){
  foreach(j=1:4, .combine=cbind) %do%{
    quantile(par[,j],c(0.025,0.975))}
}

##get the 95% test performance for all paramters 
auc_ci = quantile(auc_final_c,c(0.025, 0.975))
##99%, 95%,90% and 75%
all_ci = lapply(all_per,quar)

all_table = foreach(i = 1:4)%do%{
 paste0(round(all_per0[[i]],4)*100,'% (') %>% paste0(round(all_ci[[i]][1,],4)*100,'%, ') %>%
      paste0(round(all_ci[[i]][2,],4) *100, '%)')
}
 all_table2 = data.frame(matrix(unlist(all_table), nrow=length(all_table), byrow=F))
 rownames(all_table2) <- c(">99%", ">95%", ">90%", ">75%")
 colnames(all_table2) <- c( "Sensitivity (95% CI)", "Specificity (95% CI)",
                            "PPV (95% CI)", "NPV (95% CI)")
 knitr::kable(all_table2, 
              caption="Classification accuracy ofr suicide risk with 95% CI")
 
 ####save all predictions 
 ## for the validation\testing set 
 auc_final_testing = c(auc_final,auc_ci)
 save(auc_final_testing, file = 'auc_testing_split.Rdata')
 save(all_table2, file = 'test_perf_split.RData')

 
 
 
 
 
 
 
