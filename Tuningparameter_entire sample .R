library(tidyverse)
library(doParallel)
library(bit64)
library(ranger) #random forest package
library(sqldf)
library(data.table)
library(cvAUC)
library(foreach)


##read the data in 
load('development.RData')
##--------------------------------------- Step 1 --------------------------------------
#fit the model 0 on the original set 
data_s = dev %>% arrange(cvvar) %>% data.table()
rm(dev);gc()
#------------------Step 2:find the predictions with the best AUC 
#Tuning the parameter minimal node size in the traning set 
feat <- setdiff(colnames(data_s),c("person_id","visit_seq","death30", "death90", "training", "cvvar", "everdied"))

#specify number of trees for random forest. I expect 100 will be enough
cur.num.trees <- 100 

#specify minimum node sizes over for tuning parameter selection
min.node.size <- c(50000,100000,50000, 250000,500000) 

#specify number of features to randomly sample at each split of tree
#for this dataset, the default number of predictors that random forest would select is sqrt(p)=11 (where p is number of predictors)
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
    samp.list[[ii]] <- sboot(data_s[cvvar != i,], "death90")
  }
  samp.list
}
rm(samp.list)
#this code will do 5-fold CV, it estimates random forests for each commbination of tuning parameters and saves predictions for the left-out fold

set.seed(77)
preds_cv <- foreach(j=min.node.size, .combine=cbind) %do% {   #code cycles through all options for min.node.size
  preds_nv <- foreach(i=node.vars, .combine=cbind) %do% {    #code cycles through all options for all node.vars
    preds <- foreach(k=1:5, .combine=c) %do% {  #code cylces through CV indices 1-5
      #this next function estimates a random forest model with the selected min.node.size, node.vars. and excluded CV fold
      rf_mod <- ranger(dependent.variable.name="death90",  #specify outcome to predict
                       data = data_s[cvvar != k, c(feat,"death90"), with=FALSE],  #specify predctor data (subset all training data to exclude held-out fold and select only columns with predictors)
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
      pre_cv = predict(rf_mod,data_s[cvvar == k, c(feat), with=FALSE]) #make predictions for the held-out fold from rf_mod fit above
      pre_cv$predictions[,2]  
    } #end preds
    preds ##save all predictions 
  } #end auc_nv 
  preds_nv
} #end outer foreach


rm(preds);rm(preds_nv)
rm(cv.list);rm(rf_mod)
##find the corresponding best AUC 
auc_cv = sapply(1:ncol(preds_cv), function(k){
  AUC(preds_cv[,k],data_s$death90) })
#save all AUCs from the tuning parameter combinations 
save(auc_cv, file = 'tunedauc.RData')
#once n trees and mtry, minimal node size is decided from the auc.cv 
auc_best = max(auc_cv)
l = length(node.vars)
if(which(auc_cv == auc_best) %% l == 0){
  ntry = node.vars[l]
}else{
  ntry = node.vars[which(auc_cv == auc_best) %% l]
}
minnode = min.node.size[ceiling(which(auc_cv == auc_best) / l)]
save(ntry, file = 'ntry.RData'); save(minnode, file = 'minnode.RData')

#save the predictions for future use 
preds_best = preds_cv[,which(auc_cv == max(auc_cv))]
rm(preds_cv)
save(preds_best, file = 'preds_5folds.RData') #for future sensitivity use  
rm(preds_best)
save(auc_best, file = 'AUC_cv.RData');


##------------------------finish tunning the parameter ---------------------------
print('End of parameter tuning step')








