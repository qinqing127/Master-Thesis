library(tidyverse)
library(doParallel)
library(bit64)
library(ranger) #random forest package
library(sqldf)
library(data.table)
library(cvAUC)
library(foreach)

#read the prospective dataset in 
load('prospective.RData')
#load the split model in
load('model_split.RData')
rf_finals = rf_final
#load the primary model for entire sample estimation 
load('model0.RData')
load('feat.RData')

###functions used for finding threshold estimation 
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
sboot <- function(data, outcome){
  indi <- rep(0,nrow(data))
  ind1 <- data[get(outcome)==1, which=TRUE]
  ind0 <- data[get(outcome)==0, which=TRUE]
  booti <- data.table(ind=c(sample(ind1, replace=TRUE), sample(ind0, replace=TRUE)))
  booti <- booti[, .(freq=.N), by=ind] # make a frequency summary table
  indi[booti$ind] <- booti$freq
  return(indi) ## frequency of each observation to be chosen
}

##-------for split sample 
pre_pro0s = predict(rf_finals, data = pro[,c(feat,'death90'), with = FALSE])$predictions[,2]
rm(rf_finals); gc()
##find the AUC on the testing set  
auc_s = round(AUC( pre_pro0s,labels = pro$death90),3)

#define percentiles interested in: 99, 95, 90, 75
ps<-c(0.99, 0.95, 0.9, 0.75)

#define cutoffs based on the distribution of predictions
load('preds_split.RData')
#Sensitivity and others 
all_pros = test_perf(ps,pro$death90,pre_train, pre_pro0s )
all_pro2s = data.frame(matrix(unlist(all_pros), nrow=length(all_pros), byrow=F))
all_pro2s = round(all_pro2s,3)



##----for the whole sample model
pre_pro0 = predict(rf_final, data = pro[,c(feat,'death90'), with = FALSE])$predictions[,2]
rm(rf_final);gc()
auc_w = round(AUC( pre_pro0,labels = pro$death90),3)
load('preds_whole.Rdata')
all_pro = test_perf(ps,pro$death90,pre_final,pre_pro0 )
all_pro2 = data.frame(matrix(unlist(all_pro), nrow=length(all_pro), byrow=F))
all_pro2 = round(all_pro2,3)

#------------------------------bootstrap the whole prospective dataset 
#create empty matrix and vector to save information
#function to get 95% ci for each parameter 
quar = function(par){
  foreach(j=1:4, .combine=cbind) %do%{
    quantile(par[,j],c(0.025,0.975))}
}


#For the whole prospective dataset
#create a table includes index, true outcome, predicts from split model and from whole model
tableb = cbind( pro$death90, pre_pro0s, pre_pro0)
colnames(tableb) = c('death90', 'pre_pro0s',"pre_pro0")
 tableb = data.table(tableb)
 rm(pro); rm(pre_pro0);rm(pre_pro0s)
#-------bootstrap data 
set.seed(97)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
boot_ci_pro <- foreach(b=1:250, .verbose=T) %dopar% {
  library(data.table)
  ##bootstrap sampling
  boot1 = sboot(tableb,'death90')
  prob = data.table(tableb[rep(1:nrow(tableb),boot1)])
  #--------------for split sample predictions 
  ##find the AUC on the prospective set
  library(cvAUC)
  auc_final_pro_s <-  AUC(prob$pre_pro0s,labels = prob$death90)
  
  ##calculate sensitivity, specificity, PPV and NPV 
  test_other_pro_s =  test_perf(ps,prob$death90,pre_train,prob$pre_pro0s )
  test_other_pro_s['auc_finals'] = auc_final_pro_s

  
  #------------for whole sample estimation
  ##find the AUC on the prospective set
  auc_final_pro <-  AUC(prob$pre_pro0,labels = prob$death90)
  
  ##calculate sensitivity, specificity, PPV and NPV 
  test_other_pro =  test_perf(ps,prob$death90,pre_final,prob$pre_pro0 )
  test_other_pro['auc_final'] = auc_final_pro
 list(test_other_pro_s, test_other_pro)
  }
stopCluster(cl)


#--------------for the split sample estimation 
auc_final_pros= rep(0,250)
boot_ci_pro_s = list(); boot_ci_pro_w = list()
sensitivity_pros = matrix(nrow = 250,ncol = 4);specificity_pros = matrix(nrow = 250,ncol = 4)
PPV_pros = matrix(nrow = 250,ncol = 4) ;NPV_pros = matrix(nrow = 250,ncol = 4)
#saving all paramters into corresponding listings 

for (b in 1:250){
  boot_ci_pro_s[[b]] = boot_ci_pro[[b]][1] 
  boot_ci_pro_w[[b]] = boot_ci_pro[[b]][2] 
}

for(b in 1:250){
  auc_final_pros[b] <- unlist(boot_ci_pro_s[[b]])[17]
  sensitivity_pros[b, ] <- unlist(boot_ci_pro_s[[b]])[1:4]
  specificity_pros[b, ] <- unlist(boot_ci_pro_s[[b]])[5:8]
  PPV_pros[b, ] <- unlist(boot_ci_pro_s[[b]])[9:12]
  NPV_pros[b, ] <- unlist(boot_ci_pro_s[[b]])[13:16]
}


#make all test performance into a list(except aucaic)
all_per_pro_s = list(sensitivity_c = sensitivity_pros, specificity_c = specificity_pros,
                   PPV_c = PPV_pros,NPV_c = NPV_pros)
##get the 95% test performance for all paramters 
auc_ci_pro_s = quantile(auc_final_pros,c(0.025, 0.975))
##99%, 95%,90% and 75%
all_ci_pro = lapply(all_per_pro_s,quar)


all_table_pro_s = foreach(i = 1:4)%do%{
  paste0(round(all_pros[[i]],4)*100,'% (')%>% paste0(round(all_ci_pro[[i]][1,],4)*100,'% ,') %>%
    paste0(round(all_ci_pro[[i]][2,],4) *100, '%)')
}
all_table2_pro_s = data.frame(matrix(unlist(all_table_pro_s), nrow=length(all_table_pro_s), byrow=F))
rownames(all_table2_pro_s) = c('>99%','>95%','>90%','>75%')
colnames(all_table2_pro_s) = c('Sensitivity', 'Specificity', 'PPV','NPV')


#---------------for the whole sample estimation 
#create empty matrix and vector to save information
#--------------for the split sample estimation 
auc_final_prow= rep(0,250)
sensitivity_prow = matrix(nrow = 250,ncol = 4);specificity_prow = matrix(nrow = 250,ncol = 4)
PPV_prow = matrix(nrow = 250,ncol = 4) ;NPV_prow = matrix(nrow = 250,ncol = 4)
#saving all paramters into corresponding listings 

for(b in 1:250){
  auc_final_prow[b] <- unlist(boot_ci_pro_w[[b]])[17]
  sensitivity_prow[b, ] <- unlist(boot_ci_pro_w[[b]])[1:4]
  specificity_prow[b, ] <- unlist(boot_ci_pro_w[[b]])[5:8]
  PPV_prow[b, ] <- unlist(boot_ci_pro_w[[b]])[9:12]
  NPV_prow[b, ] <- unlist(boot_ci_pro_w[[b]])[13:16]
}


#make all test performance into a list(except aucaic)
all_per_pro_w = list(sensitivity_c = sensitivity_prow, specificity_c = specificity_prow,
                     PPV_c = PPV_prow,NPV_c = NPV_prow)
##get the 95% test performance for all paramters 
auc_ci_pro_w = quantile(auc_final_prow,c(0.025, 0.975))
##99%, 95%,90% and 75%
all_ci_pro_w = lapply(all_per_pro_w,quar)


all_table_pro_w = foreach(i = 1:4)%do%{
  paste0(round(all_pro2[[i]],4)*100,'% (')%>% paste0(round(all_ci_pro_w[[i]][1,],4)*100,'% ,') %>%
    paste0(round(all_ci_pro_w[[i]][2,],4) *100, '%)')
}
all_table2_pro_w = data.frame(matrix(unlist(all_table_pro_w), nrow=length(all_table_pro_w), byrow=F))
rownames(all_table2_pro_w) = c('>99%','>95%','>90%','>75%')
colnames(all_table2_pro_w) = c('Sensitivity', 'Specificity', 'PPV','NPV')

save(all_table2_pro_s, file = 'test_perf_pro_split.RData')
save(all_table2_pro_w, file = 'test_perf_pro_whole.RData')
auc_s = c(auc_s, auc_ci_pro_s)
save( auc_s, file = 'auc_pro_split.RData')
auc_w = c(auc_w, auc_ci_pro_w);save( auc_w, file = 'auc_pro_whole.RData')
