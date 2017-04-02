library("rpart")
library("rattle")
library("rpart.plot")
library("RColorBrewer")
library("mice")
library("VIM")
library(caret)
##recursive partitioning and regression trees
input_dataset <- "C:/Users/jennyc.wang/Downloads/R Task/titanic.csv"
predict_dataset <- "C:/Users/jennyc.wang/Downloads/R Task/predict.csv"

input_df <- read.csv(input_dataset , stringsAsFactors = F )
predict_df <- read.csv(predict_dataset)

#combine them
predict_df$Survived <- NA
predict_df <- predict_df[ , names(input_df)]
df_cmb <- rbind( input_df , predict_df )

###--------------input missing value------------------
#train.mis <- subset( train , select = -c(Name , Sex , Embarked ))
#md.pattern( train.mis )
#mice_plot <- aggr( train.mis , col=c( "navyblue", "yellow") , 
#                   numbers = TRUE , sortVars = TRUE , 
#                   labels = names(train.mis) , cex.axis = .7, 
#                   gap = 3 , ylab=c( "Missing data" , "Pattern" ))
#imputed_data <- mice( train.mis , m = 5 , maxit = 50 , method = 'pmm' , seed = 500 )
#complete_data <- complete( imputed_data , 2 )
#---------------------------------------------------------

##Name split
df_cmb$Title <- sapply( df_cmb$Name , FUN = function(x) {strsplit( df_cmb$Name , '[,[:space:]]')[[1]][3]} )
df_cmb$FamilyName <- sapply( df_cmb$Name , FUN = function(x) { strsplit( x, split = '[,.]')[[1]][1]})
df_cmb$FamilySize <- df_cmb$SibSp + df_cmb$Parch + 1 
df_cmb$FamilyID <- paste(as.character(df_cmb$FamilySize) , df_cmb$FamilyName , sep = "")
famIds <- data.frame( table( df_cmb$FamilyID ))
df_cmb$FamilyID[ df_cmb$FamilySize <= 2 ] <- 'Small'
df_cmb$FamilyID <- factor( df_cmb$FamilyID )

#Fare to idenitfy location
fareIds <- unique( df_cmb[ , c( "Fare", "Pclass" )])
fareIds_mean <- aggregate( Fare~Pclass , fareIds , mean )
names(fareIds_mean) <- c( "Pclass", "meanFare")
df_mergeedMeanFare <- merge( df_cmb , fareIds_mean, by = "Pclass")
df_mergeedMeanFare$FareDiff <- df_mergeedMeanFare$Fare - df_mergeedMeanFare$meanFare

#clean Embarked
df_mergeedMeanFare$Embarked[ which( is.na( df_mergeedMeanFare$Embarked ) ) ] <- ""

Actual_datasets <- df_mergeedMeanFare[ -which(is.na(df_mergeedMeanFare$Survived)), ]
Predict_datasets <- df_mergeedMeanFare[ which(is.na(df_mergeedMeanFare$Survived)), ]

sampling_count <- 50 
impute_data_counts <- 5 
correct_percent_list <- matrix( , nrow = sampling_count , ncol = impute_data_counts )
var_importance_matrix <- matrix( , nrow = sampling_count , ncol = impute_data_counts )
confusion_matrix <- matrix( , nrow = 2 , ncol = 2 )
confusion_Matrix_matrix <- matrix( , nrow = sampling_count , ncol = impute_data_counts )
###############################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
for ( i in  1 : sampling_count ){

train <- Actual_datasets[ sample(nrow(Actual_datasets) , as.integer( nrow(Actual_datasets) * 0.7 ) ), ]
test <- Actual_datasets [ - which( Actual_datasets$PassengerId %in% train$PassengerId ) , ]
#------------------------------------------------
train.mis <- subset( train , select = -c(Name , Sex , Embarked ))
imputed_data <- mice( train.mis , m = impute_data_counts , maxit = 50 , method = 'pmm' , seed = 500 )

for( j in 1 : impute_data_counts ){
complete_data <- complete( imputed_data , j )
train_v2 <- merge( complete_data , train[ , c( "PassengerId", "Name", "Sex" , "Embarked" )] , by = "PassengerId" )
#Generate age class
train_v2$Age <- as.numeric( train_v2$Age )
train_v2[ which( train_v2$Age < 10 ), 'AgeClass'] <- "Kid"
train_v2[ which( train_v2$Age >= 10 & train_v2$Age < 18 ), 'AgeClass'] <- "Youth"
train_v2[ which( train_v2$Age >= 18 & train_v2$Age < 65 ), 'AgeClass'] <- "Adult"
train_v2[ which( train_v2$Age >=65 ), 'AgeClass'] <- "Senior"
#test
test$Age <- as.numeric( test$Age )
test[ which( test$Age < 10 ), 'AgeClass'] <- "Kid"
test[ which( test$Age >= 10 & test$Age < 18 ), 'AgeClass'] <- "Youth"
test[ which( test$Age >= 18 & test$Age < 65 ), 'AgeClass'] <- "Adult"
test[ which( test$Age >=65 ), 'AgeClass'] <- "Senior"

fit <- rpart( Survived ~ Pclass + Sex + AgeClass + SibSp + Parch + Fare + Embarked + FamilyID + FareDiff,
              data = train_v2 , method = "class" )

var_importance_matrix[i,j] <- varImp(fit)

fancyRpartPlot(fit)

Prediction <- predict( fit, test , type = "class" )
submit <- data.frame( PassengerId = test$PassengerId , Survived = Prediction )

##
df_merge <- merge( test , submit , by = 'PassengerId' )
##confusion matrix
confusion_matrix[1,1] <- length( which( df_merge$Survived.x == 1 & df_merge$Survived.y == 1 ) )/ nrow( df_merge )
confusion_matrix[1,2] <- length( which( df_merge$Survived.x == 0 & df_merge$Survived.y == 1 ) )/ nrow( df_merge )
confusion_matrix[2,1] <- length( which( df_merge$Survived.x == 1 & df_merge$Survived.y == 0 ) )/ nrow( df_merge )
confusion_matrix[2,2] <- length( which( df_merge$Survived.x == 0 & df_merge$Survived.y == 0 ) )/ nrow( df_merge )
confusion_matrix_df <- as.data.frame(confusion_matrix)
names(confusion_matrix_df) <- c( "Survived(Test)", "Not Survived(Test)" )
rownames(confusion_matrix_df) <- c( "Survived(Train)" , "Not Survived(Train)" )
confusion_Matrix_matrix[ i , j ] <- confusion_matrix_df

correct_percentage <- length( which( df_merge$Survived.y == df_merge$Survived.x ) ) / nrow( df_merge )
correct_percent_list[i,j] <- correct_percentage

}
}

#assign 0 to Na
correct_percent_list[ is.na(correct_percent_list) ] <- 0
#convert matrix to unlist
unlst_correction <- unlist( as.list( as.data.frame( correct_percent_list )))
#mean
correct_mean <- mean( setdiff( unlst_correction , 0 ) )
std_devi <- sd( setdiff( unlst_correction , 0 ) , na.rm = F )
###################################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

actual.mis <- subset( Actual_datasets , select = -c(Name , Sex , Embarked ))
imputed_actual_data <- mice( actual.mis , m = impute_data_counts , maxit = 50 , method = 'pmm' , seed = 500 )

submit_list <- list()
var_importance <- list()
for( j in 1 : impute_data_counts ){
  complete_actual_data <- complete( imputed_actual_data , j )
  actual_v2 <- merge( complete_actual_data , Actual_datasets[ , c( "PassengerId", "Name", "Sex" , "Embarked" )] , by = "PassengerId" )
  #Generate age class
  actual_v2$Age <- as.numeric( actual_v2$Age )
  actual_v2[ which( actual_v2$Age < 10 ), 'AgeClass'] <- "Kid"
  actual_v2[ which( actual_v2$Age >= 10 & actual_v2$Age < 18 ), 'AgeClass'] <- "Youth"
  actual_v2[ which( actual_v2$Age >= 18 & actual_v2$Age < 65 ), 'AgeClass'] <- "Adult"
  actual_v2[ which( actual_v2$Age >=65 ), 'AgeClass'] <- "Senior"
  #test
  Predict_datasets$Age <- as.numeric( Predict_datasets$Age )
  Predict_datasets[ which( Predict_datasets$Age < 10 ), 'AgeClass'] <- "Kid"
  Predict_datasets[ which( Predict_datasets$Age >= 10 & Predict_datasets$Age < 18 ), 'AgeClass'] <- "Youth"
  Predict_datasets[ which( Predict_datasets$Age >= 18 & Predict_datasets$Age < 65 ), 'AgeClass'] <- "Adult"
  Predict_datasets[ which( Predict_datasets$Age >=65 ), 'AgeClass'] <- "Senior"
  
  fit_real <- rpart( Survived ~ Pclass + Sex + AgeClass + SibSp + Parch + Fare + Embarked + FamilyID + FareDiff,
                data = actual_v2 , method = "class" )
  var_importance[[j]] <- varImp(fit_real)
  fancyRpartPlot(fit_real)
  
  Prediction_real <- predict( fit_real , Predict_datasets , type = "class" )
  submit_real <- data.frame( PassengerId = Predict_datasets$PassengerId , Survived = Prediction_real )
  submit_list[[j]] <- submit_real
}

#compare 5 outcomes
for( x in 1 : impute_data_counts ){ names(submit_list[[x]])[2] <- paste0( "Survived-", x ) }
df_combined <- as.data.frame ( Reduce( function(x,y) { merge(x , y, by = "PassengerId" ) }, submit_list ) )
df_combined <- as.data.frame( lapply( df_combined , function(x) as.numeric(as.character(x)) ) )
df_combined$vari <- rowSums( df_combined[ , grep( 'survived' , names(df_combined) , ignore.case = T )])
table(df_combined$vari)
#Final(anyone is ok)
Final <- merge( predict_df , df_combined[ , c("PassengerId","Survived.1")] , by = "PassengerId" )
Final$Survived <- Final$Survived.1
Final <- Final[ , names(predict_df)]
write.csv(Final , file = "C:/Users/jennyc.wang/Downloads/R Task/predictv2.0.csv" , row.names = F )
