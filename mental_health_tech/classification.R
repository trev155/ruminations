# Importing the dataset

original_dataset = read.csv("mental_health_in_tech_survey.csv")

# -- Filter columns of interest
dataset = original_dataset[original_dataset$Country == "United States",]
dataset = dataset[dataset$tech_company == "Yes",]

cols <- c("Age", "Gender", "no_employees", "coworkers", "treatment", "mental_health_consequence")
dataset = dataset[cols]

# -- Clean data
# clean age data
median_age = median(dataset[["Age"]])
age_adjust <- function(a, replacement) {
  if (a < 0) {
    print(a)
    return(replacement)
  }
  if (a > 100) {
    print(a)
    return(replacement)
  }
  return(a)
}
dataset["Age"] = apply(dataset["Age"], 1, age_adjust, replacement=median_age)

# clean gender data
decipher_gender <- function(s) {
  s = tolower(s)
  if (s %in% c("m", "male", "man", "cis male", "male (cis)", "make", "cis man", "malr", "msle")) {
    return("m")
  } else if (s %in% c("f", "female", "woman", "female (cis)", "femake")) {
    return("f")
  } else {
    return("o")
  }
}
dataset["Gender"] = apply(dataset["Gender"], 1, decipher_gender)

# encode dependent variable as factor
dataset$mental_health_consequence = factor(dataset$mental_health_consequence, levels=c("Yes", "No", "Maybe"), labels=c(1, 2, 3))

# encode features as factors
dataset$Gender = factor(dataset$Gender, levels=c("m", "f", "o"), labels=c(1, 2, 3))
dataset$no_employees = factor(dataset$no_employees, levels=c("1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"), labels=c(1,2,3,4,5,6))
dataset$coworkers = factor(dataset$coworkers, levels=c("Yes", "No", "Some of them"), labels=c(1,2,3))
dataset$treatment = factor(dataset$treatment, levels=c("Yes", "No"), labels=c(1,2))


# -- Building a model for the data
# TODO - this isn't a good model
# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$mental_health_consequence, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# fit classifier
library(randomForest)
classifier = randomForest(x = training_set[1:5], y = training_set$mental_health_consequence, ntree = 1000)

# predict test results
y_pred = predict(classifier, newdata = test_set)

# confusion matrix
cm = table(test_set$mental_health_consequence, y_pred)

