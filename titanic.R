# =========================================================================================== #
# library
# =========================================================================================== #
library(randomForest)

# =========================================================================================== #
# load train, competition data and creation of the full dataset
# =========================================================================================== #

path <- getwd()
titanic.competition <- read.csv(file = paste0(path,"/test.csv"),
                         stringsAsFactors = FALSE,
                         header = TRUE
)
titanic.competition$dataType <- "competition"
titanic.competition$Survived <- NA

titanic.data <- read.csv(file = paste0(path,"/train.csv",
                          stringsAsFactors = FALSE,
                          header = TRUE
)
titanic.data$dataType <- "data"

ind <- sample(2, nrow(titanic.data), replace = TRUE, prob = c(0.7, 0.3))
titanic.train <- titanic.data[ind==1,]
titanic.test  <- titanic.data[ind==2,]

titanic.full <- rbind(titanic.train, titanic.test, titanic.competition)

# =========================================================================================== #
# data preparation
# =========================================================================================== #

table(titanic.train$Embarked)
titanic.train$Embarked[titanic.train$Embarked == ''] <- 'S'
table(titanic.train$Embarked)

table(titanic.test$Embarked)
titanic.test$Embarked[titanic.test$Embarked == ''] <- 'S'
table(titanic.test$Embarked)

table(titanic.competition$Embarked)
titanic.competition$Embarked[titanic.competition$Embarked == ''] <- 'S'
table(titanic.competition$Embarked)
#
table(is.na(titanic.train$Age))
titanic.train$Age[is.na(titanic.train$Age)]
age.median <- median(titanic.train$Age, na.rm = TRUE)
titanic.train$Age[is.na(titanic.train$Age)] <- age.median
table(is.na(titanic.train$Age))
# titanic.train$Age[titanic.train$Age >= 27] <- 50 
# titanic.train$Age[titanic.train$Age < 27]  <- 1 

table(is.na(titanic.test$Age))
titanic.test$Age[is.na(titanic.test$Age)]
age.median <- median(titanic.test$Age, na.rm = TRUE)
titanic.test$Age[is.na(titanic.test$Age)] <- age.median
table(is.na(titanic.test$Age))
# titanic.test$Age[titanic.test$Age >= 27] <- 50 
# titanic.test$Age[titanic.test$Age < 27]  <- 1 

table(is.na(titanic.competition$Age))
titanic.competition$Age[is.na(titanic.competition$Age)]
age.median <- median(titanic.competition$Age, na.rm = TRUE)
titanic.competition$Age[is.na(titanic.competition$Age)] <- age.median
table(is.na(titanic.competition$Age))
# titanic.competition$Age[titanic.competition$Age >= 27] <- 50 
# titanic.competition$Age[titanic.competition$Age < 27]  <- 1 
#
table(is.na(titanic.train$Fare))
fare.median <- median(titanic.train$Fare, na.rm = TRUE)
titanic.train$Fare[is.na(titanic.train$Fare)] <- 
table(is.na(titanic.train$Fare))

table(is.na(titanic.test$Fare))
fare.median <- median(titanic.test$Fare, na.rm = TRUE)
titanic.test$Fare[is.na(titanic.test$Fare)] <- fare.median
table(is.na(titanic.test$Fare))

table(is.na(titanic.competition$Fare))
fare.median <- median(titanic.competition$Fare, na.rm = TRUE)
titanic.competition$Fare[is.na(titanic.competition$Fare)] <- fare.median
table(is.na(titanic.competition$Fare))
#
titanic.train$Pclass     <- as.factor(titanic.train$Pclass)
titanic.train$Sex        <- as.factor(titanic.train$Sex)
titanic.train$Embarked   <- as.factor(titanic.train$Embarked)
titanic.train$Survived   <- as.factor(titanic.train$Survived)
# titanic.train$Age        <- as.factor(titanic.train$Age)

titanic.test$Pclass     <- as.factor(titanic.test$Pclass)
titanic.test$Sex        <- as.factor(titanic.test$Sex)
titanic.test$Embarked   <- as.factor(titanic.test$Embarked)
titanic.test$Survived   <- as.factor(titanic.test$Survived)
# titanic.test$Age        <- as.factor(titanic.test$Age)

titanic.competition$Pclass     <- as.factor(titanic.competition$Pclass)
titanic.competition$Sex        <- as.factor(titanic.competition$Sex)
titanic.competition$Embarked   <- as.factor(titanic.competition$Embarked)
titanic.competition$Survived   <- as.factor(titanic.competition$Survived)
# titanic.competition$Age        <- as.factor(titanic.competition$Age)

# =========================================================================================== #
# tuning and model construction
# =========================================================================================== #

t <- tuneRF(titanic.train[,c(-1,-2,-4,-9,-11,-13)], titanic.train[,2],
            stepFactor = 2.5,
            plot = TRUE,
            ntreeTry = 500,
            trace = TRUE,
            improve = 0.05)

titanic.model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked +
                                  Pclass*Sex + Age*SibSp + Parch*Fare + 
                                  Sex*Age + SibSp*Parch + Fare*Embarked +
                                  Age*Pclass, 
                                  data = titanic.train,
                                  mtree = 500,
                                  mtry = 2,
                                  nodesize = 0.01*nrow(titanic.train)
)
plot(titanic.model)

# =========================================================================================== #
# validation and prevision
# =========================================================================================== #

Survived.test <- predict(titanic.model, newdata = titanic.test)
previsione <- table(Survived.test, titanic.test[,2])
(previsione[1,1]+previsione[2,2])/sum(previsione)

# =========================================================================================== #
# competition result
# =========================================================================================== #

Survived <- predict(titanic.model, newdata = titanic.competition)
PassengerId <- titanic.competition$PassengerId

result.df <- data.frame(PassengerId = PassengerId, Survived = as.integer(Survived)-1)
str(result.df)

write.csv(result.df, file = paste0(path,"/kaggle_sub.csv", row.names = FALSE)

