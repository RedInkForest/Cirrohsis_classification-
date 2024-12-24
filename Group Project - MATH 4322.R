##### Group Project - Group 6 - Cirrhosis Prediction

# Load necessary library:
library(ggplot2)
library(corrplot)
library(GGally)
library(tree)

## Read the data set:
cirrhosis <- read.csv("C:/Users/Hung/Desktop/cirrhosis.csv")


## Data exploration:
  # Summary of the data set:
head(cirrhosis, 10)
summary(cirrhosis)

  # Number of observations for each Stage:
table(cirrhosis$Stage)


## Data Preprocess: (Le Bui)

  # Exclude "ID", "N_Days", "Status", and "Drug" for modeling:
cleaned_data <- cirrhosis[, -c(1:4)]

  # Age is in days, so we transform it into years:
cleaned_data$Age <- floor(cleaned_data$Age / 365)

  # Transform Stage into 2-class Early and Late:
cleaned_data$Stage <- ifelse(cleaned_data$Stage <= 2, "Early", "Late")

  # Clean NAs:
cleaned_data <- na.omit(cleaned_data)

  # Make categorical variables into factor:
cleaned_data$Sex <- as.factor(cleaned_data$Sex)
cleaned_data$Ascites <- as.factor(cleaned_data$Ascites)
cleaned_data$Hepatomegaly <- as.factor(cleaned_data$Hepatomegaly)
cleaned_data$Spiders <- as.factor(cleaned_data$Spiders)
cleaned_data$Edema <- as.factor(cleaned_data$Edema)

  # Make response variable into factor:
cleaned_data$Stage <- as.factor(cleaned_data$Stage)

  # A summary of cleaned data:
summary(cleaned_data)

## Data visualization: (Le Bui, Khoi Phan)
  # Correlation heat map between qualiative variables:
numerical_data <- cleaned_data[, c("Bilirubin", "Cholesterol", "Albumin", "Copper", 
                                   "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", 
                                   "Prothrombin")]
cormat = round(cor(numerical_data),2)
corrplot::corrplot(cormat,method = "circle",type = "upper")
    # Bilirubin has a moderate postive relationships with Copper, SGOT, Trigliceries, and Prothrombin.

  # Create pairwise scatterplots
ggpairs(numerical_data, 
        title = "Pairwise Scatterplots of Features",
        lower = list(continuous = wrap("points", alpha = 0.5, size = 0.5)),
        diag = list(continuous = "densityDiag"),
        upper = list(continuous = "cor"))

  # Create a bar plot to show the count of Hepatomegaly in each Stage
ggplot(cleaned_data, aes(x = Stage, fill = Hepatomegaly)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Hepatomegaly by Stage",
       x = "Stage",
       y = "Count",
       fill = "Hepatomegaly") +
  theme_minimal()
    # Hepatomegaly tends to be significant in Late stage.

  # Create a boxplot for Copper by Stage
ggplot(cleaned_data, aes(x = Stage, y = Copper, fill = Stage)) +
  geom_boxplot() +
  labs(title = "Distribution of Copper by Stage",
       x = "Stage",
       y = "Copper Level") +
  theme_minimal() +
  scale_fill_manual(values = c("Early" = "skyblue", "Late" = "coral"))
    # Copper level tends to increase in Late Stage.

  # Create a grouped bar plot for Sex by Stage
ggplot(cleaned_data, aes(x = Stage, fill = Sex)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Sex by Stage",
       x = "Stage",
       y = "Count",
       fill = "Sex") +
  theme_minimal()
    # Female has more observations of cirrhosis in either Early or Late stages.


  # Create a grouped bar plot for Ascites by Stage
ggplot(cleaned_data, aes(x = Stage, fill = Ascites)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Ascites by Stage",
       x = "Stage",
       y = "Count",
       fill = "Ascites") +
  theme_minimal()
    # Ascites variable is not recorded in Early Stage.
    # However, this can lead to perfect separation issue when trying with logistic regression model.

  # Histogram for Albumin
ggplot(cleaned_data, aes(x = Albumin)) +
  geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Albumin", x = "Albumin", y = "Frequency") +
  theme_minimal()
    # Albumin variable tends to skew left, but partly has a bell shape.

  # Histogram for Triglycerides
ggplot(cleaned_data, aes(x = Tryglicerides)) +
  geom_histogram(binwidth = 10, fill = "coral", color = "black", alpha = 0.7) +
  labs(title = "Histogram of Triglycerides", x = "Triglycerides", y = "Frequency") +
  theme_minimal()
    # Triglycerides variable tends to skew right.




### Apply model:

## Single Decision Tree model: (Le Bui, Khoi Phan)

library(tree)

# Fit the decision tree model using Validation Set Approach - splitting data into 80% training and 20% testing:
set.seed(6)

train_index = sample(1:nrow(cleaned_data), nrow(cleaned_data) * 0.8)

train_data_unique <- cleaned_data[train_index, ]
test_data_unique <- cleaned_data[-train_index, ]

tree_model <- tree(Stage ~ ., data = train_data_unique)

# View the model summary
summary(tree_model)

# Find the test error of the original tree:
test_tree_model = predict(tree_model, newdata = test_data_unique, type = "class")
table(test_tree_model, test_data_unique$Stage)
mean(test_tree_model != test_data_unique$Stage)


# Plot the tree
plot(tree_model)
text(tree_model, pretty = 0)

# Evaluate the model using cross-validation for potential pruning
set.seed(6)
cv_tree <- cv.tree(tree_model, FUN = prune.misclass)

# Plot cross-validation results to determine optimal tree size
plot(cv_tree$size, cv_tree$dev, type = "b", xlab = "Tree Size", ylab = "Misclassification Deviance")

# Prune the tree based on the optimal size found in cross-validation (adjust 'best' value as needed)
optimal_size <- cv_tree$size[which.min(cv_tree$dev)]
pruned_tree <- prune.misclass(tree_model, best = optimal_size)
summary(pruned_tree)


# Plot the pruned tree
plot(pruned_tree, main = "Plot of the pruned tree")
text(pruned_tree, pretty = 0)

# See test result from pruned tree:
test_pruned_tree_model = predict(pruned_tree, newdata = cleaned_data[-train_index, ], type = "class")
table(test_pruned_tree_model, cleaned_data[-train_index, "Stage"])
mean(test_pruned_tree_model != cleaned_data[-train_index, "Stage"] )




# Split the data 10 times and obtain the test results:

test_error_tree <- numeric(10)

for (i in 1:10) {
  set.seed(i*5) # set different seeds
  
  train = sample(1:nrow(cleaned_data), nrow(cleaned_data) * 0.8)
  train_data = cleaned_data[train, ]
  test_data = cleaned_data[-train, ]
                # randomly split the data into 80% training and 20% testing
  
  # Apply tree model
  tree_model <- tree(Stage ~ ., data = train_data)
  
  # Find best size by CV:
  cv_tree <- cv.tree(tree_model, FUN = prune.misclass)
  
  # Filter for sizes greater than or equal to 2 and find the one with the lowest deviance
  valid_sizes <- cv_tree$size[cv_tree$size >= 2]
  valid_devs <- cv_tree$dev[cv_tree$size >= 2]
  
  optimal_size <- valid_sizes[which.min(valid_devs)]
  
  pruned.tree <- prune.misclass(tree_model, best = optimal_size)
  
  cirrhosis_pruned_tree_pred = predict(pruned.tree, newdata = test_data, type = "class")

  # test error rate for pruned tree for model evaluation
  test_error_tree[i] <- mean(cirrhosis_pruned_tree_pred != test_data$Stage)
}

# Average test error rate
print(mean(test_error_tree))
summary(test_error_tree)
  # The test errors from decision trees range from 0.2143 to 0.3929, which is a wide range.
  # The average test error is 0.3054.

# Use Random Forest and VarImplot to find significant predictors:
library(randomForest)

rf.cirrhosis = randomForest(Stage ~ ., data = cleaned_data, mtry = sqrt((ncol(cleaned_data) - 1)), importance = T)
summary(rf.cirrhosis)

varImpPlot(rf.cirrhosis)
  # Hepatomegaly and Copper are two of the most significant predictors.




## Fit the logistic regression model: (Thien Ngo, Ethan Pradhan)

# We reuse train_data_unique and test_data_unique from Decision Tree part
fit.bc_all <- glm(Stage ~ ., 
              family = "binomial", data = train_data_unique)
  # There is an issue occured when fitting the logistic regression model using full predictors.
  # As explored from above, Ascites variable has the perfect separation issue.

# Fit the logistic regression using all predictors, except Ascites:

fit.bc <- glm(Stage ~ . -Ascites, 
              family = "binomial", data = train_data_unique)

summary(fit.bc)


# Check assumptions:
par(mfrow = c(2,2))
plot(fit.bc)
  # The data set does not obtain linearity and equal variance, but obtains some normality.

# Fit the logistic regression model with the test set:
percent.bc <- predict.glm(fit.bc, newdata = test_data_unique, type = "response")
predict.bc = ifelse(percent.bc <= 0.5, 'Early', 'Late')
(conf.bc <- table(Predicted = predict.bc, Actual = test_data_unique$Stage))

# Test error of the test set:
mean(predict.bc != test_data_unique$Stage)

# Find the best subset of predictors:
# Stepwise selection:
step.logistic = step(fit.bc, direction = 'both')
summary(step.logistic)
  # Hepatomegaly and Copper have a significant small p-values.
  # P-value of Spiders is slighly more than 0.05, but in comparison to other variables, it is much lower.
  # Therefore, Hepatomegaly, Copper, and Spiders are considered to be the best subset of predictors.

# We choose three best predictors with significant of p-values:
step.model = glm(Stage ~ Hepatomegaly + Copper + Spiders, family='binomial', data = train_data_unique)
summary(step.model)

percent.sw <- predict.glm(step.model, newdata = test_data_unique, type = "response")
percent.sw = ifelse(percent.sw <= 0.5, 'Early', 'Late')
(conf.bc <- table(Predicted = percent.sw, Actual = test_data_unique$Stage))
mean(percent.sw != test_data_unique$Stage)


# Loop to calculate test error rates for 10 different splits
# Initialize a vector to store test error rates
test_error_logistic <- numeric(10)
for (i in 1:10) {
  set.seed(i*5) # set different seeds
  
  train = sample(1:nrow(cleaned_data), nrow(cleaned_data) * 0.8)
  train_data = cleaned_data[train, ]
  test_data = cleaned_data[-train, ]
  # randomly split the data into 80% training and 20% testing
  
  # Apply the Logistic model:
  logistic <- glm(Stage ~ Hepatomegaly + Copper + Spiders, family='binomial', data = train_data)
  
  logistic_perc = predict(logistic, newdata = test_data, type = "response")
  logistic_pred <- ifelse(logistic_perc <= 0.5, 'Early', 'Late')
  # test error rate for pruned tree for model evaluation
  test_error_logistic[i] <- mean(logistic_pred != test_data$Stage)
}

# Print the 10 test error rates
print(test_error_logistic)
summary(test_error_logistic)

# Calculate and print the average test error rate
mean(test_error_logistic)




