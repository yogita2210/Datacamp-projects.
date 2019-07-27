###1## Loading Libraries and Data:

##1.1)
library(dplyr)
library(ggplot2)
library('ggthemes') 
library('scales')
library('mice')

library('rpart')
library('rpart.plot')
library('caret')
library('randomForest') 


##1.2) Cleaning workspace
#removing all objects/variables from global env to free memory space

rm(list=ls())

##1.3) Loading data as df

test<-read.csv("../input/test.csv",stringsAsFactors = FALSE)

training<-read.csv("../input/train.csv",stringsAsFactors = FALSE)

##1.4) examining the dataframe
class(test)
str(test)
summary(test)

dim(test)
head(test)
colnames(test)



class(training)
str(training)
summary(training)

dim(training)
head(training)
colnames(training)


##1.5## Combining test and training datset:

# making survived column in test dataset, with NA values(will help in combining the 2 datasets)

test$Survived<-rep(NA,nrow(test))

##binding training and test data together

titanic_data<-rbind(training,test)

# check dtype using class()
class(titanic_data)

#checking if test and train data are combined to form titanic_data

nrow(titanic_data)==nrow(test)+nrow(training)

###2## Pre-Processing the Data:

#2.1 #Missing values

#2.1.1) Column wise, no. of missing and empty values:

colSums(is.na(titanic_data))
#Survived have 418 NA values(test dataset),Fare have 1,Age have 263

#olSums(titanic_data=="")
#Cabin column have 1014 empty values, embarked have 2
##also Survived and Age is giving NA, as they contain, to evaulate " " in columns with NA

#colSums(titanic_data=="",na.rm=TRUE) - will give no. of empty values, excluding NA.

##2.2##Dealing with Missing Values

#2.2.1#Embarked

#first replace " " with NA
 titanic_data$Embarked[titanic_data$Embarked == ""] <- NA
    
## To know which passengers have no listed embarkment port


titanic_data[(which(is.na(titanic_data$Embarked))), 1] 

paste(replicate(50, "<>"), collapse = "")
#Give the TRUE indices of a logical object, then we will use df[rowindice,columnindice(1,i.e passenger ID)]
#will give us row index, where na is present

##cross_validating, by evaluating  the value of embarked at given row index
titanic_data[c(62, 830), 'Embarked'] 
## So Passenger numbers 62 and 830 are each missing their embarkment ports.

#now, we will explore feature of abv 2 features, in order to check if there is anythng common bw 2 passengers
## Let's look at their class of ticket and their fare. 

titanic_data[c(62, 830), c(1,3,10)] 

## Both passengers had first class tickets that they spent 80 (pounds?) on. 

## Let's see the embarkment ports of others who bought similar kinds of tickets. 


## first way of handling missing value in embarked 
#titanic_data[c(62, 830), c(1,3,10)] -can't use this method, to extract value as we need to apply cndn.

(titanic_data%>% 
  group_by(Embarked, Pclass) %>% 
    filter(Pclass == "1") %>% 
     summarise(mfare = median(Fare),n = n()))
## Looks like the median price for a first class ticket departing from 'C' ## (Charbourg) was 77 (in comparison to our 80).
#While first class tickets ## departing from 'Q' were only slightly more expensive (median price 90), ## only 3 first class passengers departed from that port. It seems far 
## more likely that passengers 62 and 830 departed with the other 141 ## first-class passengers from Charbourg.

## Second Way of handling missing value in embarked 

embark_fare <- titanic_data %>% 
 filter(PassengerId != 62 & PassengerId != 830) 


 
# Use ggplot2 to visualize embarkment, passenger class, & median fare 

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) + geom_boxplot() + 
 geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=2) + scale_y_continuous(labels=dollar_format()) + theme_few() 



## from plot we can see that The median fare for a first class passenger ## departing from Charbourg ('C') coincides nicely with the $80 paid by our ## embarkment-deficient passengers. 
#I think we can safely replace the NA values ## with 'C'. # Since their fare was $80 for 1st class, they most likely embarked from 'C' 

titanic_data$Embarked[c(62, 830)] <- 'C'

#2.2.2#Fare -1NA value

## Missing value in fare ## ## to know Which passenger has no fare information 

titanic_data[(which(is.na(titanic_data$Fare))) , 1]

 ## Looks like Passenger number 1044 has no listed Fare

 # Where did this passenger leave from? What was their class? 
titanic_data[1044, c(3, 12)] 

# Another way to know about passenger id 1044 :Show row 1044 
titanic_data[1044, ] 

## Looks like he left from 'S' (Southampton) as a 3rd class passenger. 
## Let's see what other people of the same class and embarkment port paid for ## their tickets. 
## First way: 
titanic_data%>% filter(Pclass == '3' & Embarked == 'S') %>% summarise(missing_fare = median(Fare, na.rm = TRUE)) 

## Looks like the median cost for a 3rd class passenger leaving out of ## Southampton was 8.05. That seems like a logical value for this passenger ## to have paid.

 ## Second way: 

ggplot(titanic_data[titanic_data$Pclass == '3' & titanic_data$Embarked == 'S', ], aes(x = Fare)) + geom_density(fill = '#99d6ff', alpha=0.4) + 
geom_vline(aes(xintercept=median(Fare, na.rm=T)), colour='red', linetype='dashed', lwd=1) + scale_x_continuous(labels=dollar_format()) + theme_few() 

## From this visualization, it seems quite reasonable to replace the NA Fare ## value with median for their class and embarkment which is $8.05. 

## Replace that NA with 8.05 

titanic_data$Fare[1044] <- 8.05 
summary(titanic_data$Fare) 

### Another way of Replace missing fare value with median fare for class/embarkment 

titanic_data$Fare[1044] <- median(titanic_data[titanic_data$Pclass == '3' & titanic_data$Embarked == 'S', ]$Fare, na.rm = TRUE)




 ##2.2.3## Missing Value in Age 
# Show number of missing Age values 

sum(is.na(titanic_data$Age)) 

## 263 passengers have no age listed. Taking a median age of all passengers ## doesn't seem like the best way to solve this problem, so it may be easiest to ## try to predict the passengers' age based on other known information. ## To predict missing ages, but I'm going to use the mice package. To start with ## i will factorize the factor variables and then perform ## mice(multiple imputation using chained equations


# Set a random seed 

set.seed(129)
 # Perform mice imputation, excluding certain less-than-useful variables( variables which have some correlation on with Age and which doesn't contain any missing values): 
mice_mod <- mice(titanic_data[, !names(titanic_data) %in% c('PassengerId','Name','Ticket','Cabin','Survived')], method='rf')

 # Save the complete output 

mice_output <- complete(mice_mod) 

## Let's compare the results we get with the original distribution of ## passenger ages . 

# Plot age distributions 

par(mfrow=c(1,2)) 
hist(titanic_data$Age, freq=F, main='Age: Original Data', col='darkred', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', col='lightgreen', ylim=c(0,0.04)) 


## 2 Distributions look similar, so let's replace our age vector in the original data ## with the output from the mice model.

titanic_data$Age <- mice_output$Age 

# Show new number of missing Age values 
sum(is.na(titanic_data$Age))



#2.2.4# famsize variable

##3##Visualization

#3.1# Survivial as a function of SibSp and Parch
ggplot(titanic_data[1:891,],aes(x=SibSp,fill=Survived))+geom_bar()


#3.2# Survivial as a function of Parents and children

ggplot(titanic_data[1:891,],aes(x=Parch,fill=Survived))+geom_bar()

#3.3# Let's try to look at another parameter: family size.

titanic_data$famsize <- titanic_data$SibSp + titanic_data$Parch +1;

# The dymanics of SibSp and Parch are very close one each other.

#2.2.4# famsize variable(continue)

#as we found that, SibSp and Parch are giving similar information, so lets make a new variable.
#Family size variable: We are going to create variable "famsize" to know # the number of family. it include number of sibling/number of parents ## and children+ passenger themselves 


titanic_data$famsize <- titanic_data$SibSp + titanic_data$Parch + 1 

 ## Visualize the relationship between family size & survival 

ggplot(titanic_data[1:891,], aes(x = famsize, fill = factor(Survived))) + geom_bar(stat='count', position='dodge') + scale_x_continuous(breaks=c(1:11)) + labs(x = 'Family Size') + theme_few() 

## Explanation: We can see that there's a survival penalty to single/alone and ## those with family sizes above 4.
#We can collapse this variable into three ## levels which will be helpful since there are comparatively fewer large families ## Spit variable into three levels # Discretize family size 


table(titanic_data[1:891,]$famsize)

##categorizing Family:

titanic_data$fsizeD[titanic_data$famsize == 1] <- 'single' 
titanic_data$fsizeD[titanic_data$famsize < 5 & titanic_data$famsize> 1] <- 'small'
titanic_data$fsizeD[titanic_data$famsize> 4] <- 'large'

#2.2.5# ##categorizing Age 
##3##visualization

# 3.4# Survival as a function of age:

ggplot(titanic_data[1:891,],aes(x=Age,fill=Survived))+geom_histogram(binwidth =3)

ggplot(titanic_data[1:891,],aes(x=Age,fill=Survived))+geom_histogram(binwidth = 3,position="fill")+ylab("Frequency")

#2.2.5# ##categorizing Age (Continue)
max(titanic_data$Age)
min(titanic_data$Age)


titanic_data$Agegroup<-cut(titanic_data$Age, seq(0,80,10))

# Show counts

table(titanic_data$Agegroup)

#2.2.6# ##categorizing Fare

##3##Visualization:

#3.5# Is there a correlation between Fare and Survivial?
ggplot(titanic_data[1:891,],aes(x=Fare,fill=Survived))+geom_histogram(aes(binwidth =20, position="fill"))

#2.2.6# ##categorizing Fare (continue)
max(titanic_data$Fare)
min(titanic_data$Fare)


titanic_data$Faregroup<-cut(titanic_data$Fare, seq(0,515,80))

# Show counts

table(titanic_data$Faregroup)



##2.2.7 feature engineering-name
### Retrieve title from passenger names 
titanic_data$title<-gsub('(.*, )|(\\..*)', '', titanic_data$Name)
 # Show title counts by sex 

table(titanic_data$Sex, titanic_data$title) 


## Convert title with low count into new title 
unusual_title<-c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer') 
## Rename/reassign Mlle, Ms, and Mme 

titanic_data$title[titanic_data$title=='Mlle']<-'Miss' 
titanic_data$title[titanic_data$title=='Ms']<-'Miss' 
titanic_data$title[titanic_data$title=='Mme']<-'Mrs' 



titanic_data$title[titanic_data$title %in% unusual_title]<-'Unusual Title' 
##Mr and Master have good count so will kepp them separate

## Check the title count again 

table(titanic_data$Sex, titanic_data$title) 


                             
##2.2.9 Factorizing variables


titanic_data$Pclass<-factor(titanic_data$Pclass)
titanic_data$Sex<-factor(titanic_data$Sex)
titanic_data$Embarked<-factor(titanic_data$Embarked)
titanic_data$Survived<-factor(titanic_data$Survived)
titanic_data$title<-factor(titanic_data$title)
titanic_data$fsizeD<-factor(titanic_data$fsizeD)
titanic_data$Agegroup<-factor(titanic_data$Agegroup)
titanic_data$Faregroup<-factor(titanic_data$Faregroup)
                             
paste(rep("<>",50))
##3##Visualization:

#3.6# Take a look at gross survival rates
table(titanic_data$Survived)
## 
##    0    1 
## 1098  684
paste(rep("<>",50))
#3.7# Distribution across classes
table(titanic_data$Pclass)
## 
##   1   2   3 
## 432 368 982
#class vs survival
#Hypothesis - Rich folks survived at a higer rate

ggplot(titanic_data[1:891,], aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(width = 0.5,position="fill") +
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived") 

table(titanic_data[1:891,]$Pclass)
table(titanic_data[1:891,]$Pclass,titanic_data[1:891,]$Survived)

#3.8# Sex 

# First, let's look at the relationship between sex and survival:
#training$sex<- as.factor(training$sex)

ggplot(titanic_data[1:891,], aes(x = Sex, fill = factor(Survived))) +
  geom_bar(width = 0.5) +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived") 
table(titanic_data[1:891,]$Sex)
table(titanic_data[1:891,]$Sex,titanic_data[1:891,]$Survived)

# 3.9# Survival as a function of Embarked:


titanic_data[1:891,]$Embarked <- as.factor(titanic_data[1:891,]$Embarked)
ggplot(titanic_data[1:891,], aes(x = Embarked, fill = factor(Survived))) +
  geom_bar(width = 0.5,position="fill") +
  xlab("Embarked") +
  ylab("Total Count") +
  labs(fill = "Survived") 

table(titanic_data[1:891,]$Embarked)
table(titanic_data[1:891,]$Embarked,titanic_data[1:891,]$Survived)

##3.10##
# Now, let's devide the graph of Embarked by Pclass:
ggplot(data = titanic_data[1:891,],aes(x=Embarked,fill=Survived))+geom_bar(position="fill")+facet_wrap(~Pclass)

# Now it's not so clear that there is a correlation between Embarked and Survival. 
## as for every class, the result for embarkment is different .(sometimes, its C with more survival, sometimes its Q n smtimes S)- If Embarkment point were correlated with survived, then it must 
#have same pattern for each of the class


#3.11#Title
ggplot(titanic_data[1:891,],aes(x=title,fill=Survived))+geom_bar(position="fill")+ylab("Frequency")

#3.12# survival as function of Agegroup
titanic_data[1:891,]$Agegroup <- as.factor(titanic_data[1:891,]$Agegroup)
ggplot(titanic_data[1:891,], aes(x = Agegroup, fill = factor(Survived))) +
  geom_bar(width = 0.5,position="fill") +
  xlab("Agegroup") +
  ylab("Total Count") +
  labs(fill = "Survived") 

#3.13# survival as function of Faregroup
titanic_data[1:891,]$Faregroup <- as.factor(titanic_data[1:891,]$Faregroup)
ggplot(titanic_data[1:891,], aes(x = Faregroup, fill = factor(Survived))) +
  geom_bar(width = 0.5,position="fill") +
  xlab("Faregroup") +
  ylab("Total Count") +
  labs(fill = "Survived") 


###4### Prediction


# ##4.1) Logistic Regression##



#Splitting training and test data from training data, to check accuracy of our model

# The train set with the important features 
train_im<- titanic_data [1:892,c("Survived","Pclass","Sex","Agegroup","Faregroup","fsizeD","title")]
ind<-sample(1:dim(train_im)[1],500) # Sample of 500 out of 891
train1<-train_im[ind,] # The train set of the model
test1<-train_im[-ind,] # The test set of the model




# #4.1.3# Model Creation

model <- glm(Survived ~.,family=binomial(link='logit'),data=train1) 
## Model Summary 
summary(model) 

# ## Using anova() to analyze the table of devaiance

anova(model, test="Chisq")

## Predicting Test Data
result_glm <- predict(model,newdata=test1,type='response')
result_glm <- ifelse(result_glm > 0.5,1,0)

#result_glm<-factor(result_glm)

# # Mean of the true prediction <<chk NA>>
mean(result_glm==test1$Survived,na.rm=TRUE)



t1<-table(result_glm ,test1$Survived)

# Presicion and recall of the model
presicion<- t1[1,1]/(sum(t1[1,]))
recall<- t1[1,1]/(sum(t1[,1]))
presicion
recall

# F1 score
F1<- 2*presicion*recall/(presicion+recall)
F1

# ####Applying model to orginal testing data
 testing1<-titanic_data[892:1309,c("Survived","Pclass","Sex","Agegroup","Faregroup","fsizeD","title","PassengerId")]


result_glm1 <- predict(model,newdata=testing1,type='response')
result_glm1<- ifelse(result_glm1 > 0.5,1,0)



res<- data.frame(testing1$PassengerId,result_glm1)
names(res)<-c("PassengerId","Survived")

write.csv(res,file="res.csv",row.names = F)


# # ##4.2) Decision Tree:
library(rpart)

model_dt<- rpart(Survived ~.,data=train1, method="class")
rpart.plot(model_dt)

resul_dt <- predict(model_dt,test1,type = "class")
 test1$Survived
 mean(resul_dt ==test1$Survived,na.rm=TRUE)

 t2<-table(resul_dt ,test1$Survived)

presicion_dt<- t2[1,1]/(sum(t2[1,]))
 recall_dt<- t2[1,1]/(sum(t2[,1]))

 F1_dt<- 2*presicion_dt*recall_dt/(presicion_dt+recall_dt)


 ###Applying model, to make prediction on actual testing data-testing

result_dt1 <- predict(model_dt,testing1,type="class")

 res1<- data.frame(testing1$PassengerId,result_dt1)
names(res1)<-c("PassengerId","Survived")
write.csv(res1,file="res1.csv",row.names = F)


# #4.3) Random Forest:


# # Let's try to predict survival using a random forest.
model_rf<-randomForest(Survived~.,data=train1,importance=TRUE,
        prOximity=TRUE,
         na.action=na.roughfix)

# # Let's look at the error
plot(model_rf)

result_rf <- predict(model_rf,test1)

test1$Survived
mean(result_rf ==test1$Survived,na.rm=TRUE)

 t3<-table(result_rf ,test1$Survived)

presicion_rf<- t3[1,1]/(sum(t3[1,]))
 recall_rf<- t3[1,1]/(sum(t3[,1]))

F1_rf<- 2*presicion_rf*recall_rf/(presicion_rf+recall_rf)


 ###Applying model, to make prediction on actual testing data-testing

result_rf1 <- predict(model_rf,titanic_data[892:1309,])

 res2<- data.frame(testing1$PassengerId,result_rf1)
names(res2)<-c("PassengerId","Survived")
write.csv(res2,file="res2.csv",row.names = F)


