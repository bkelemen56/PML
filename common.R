###############################################################################
# common script to be used by all other scripts.
#
###############################################################################

library(plyr)
library(dplyr)
library(ggplot2)

library(caret)
library(randomForest)

library(futile.logger)

# number or cores for parallel processing
library(doMC)
registerDoMC(cores = 6)

###############################################################################
# Initialize the process if default values are not desired
# method = rf | rpart | lda | ada
# size = size of the dataset or 0 for all
# training_size = percentage of data to use for training
# post_name = string to add to MODEL_NAME at the end (ex. trControl parameter info)
# 
# will store arguments in global variables METHOD, DATASET_SIZE, TRAINING_SIZE, 
# MODEL_NAME and set the seed for reproducibility
init <- function(method, size, training_size, post_name = "") {
  
  assign("METHOD", method, envir = .GlobalEnv) 
  assign("DATASET_SIZE", size, envir = .GlobalEnv) 
  assign("TRAINING_SIZE", training_size, envir = .GlobalEnv)

  # only add dataset size to model name if we're using a subset
  s1 <- ifelse(DATASET_SIZE == 0, "", paste0("-", as.character(DATASET_SIZE)))
  
  # only add post_name if not empty
  s2 <- ifelse(post_name == "", "", paste0("-", post_name))
  
  assign("MODEL_NAME", 
         paste0("newmodel-", METHOD, "-", as.character(TRAINING_SIZE), s1, s2), 
         envir = .GlobalEnv)
  
  # make this script reproducible
  set.seed(1956)
  
  flog.info(paste0("Processing model: ", MODEL_NAME))
  
  # return the model name
  invisible(MODEL_NAME)
}

###############################################################################
# set default parameters for each case
init_defaults <- function(model_id = "rf-100-0.7") {
  switch (model_id, 
          "rf-100-0.7" = init("rf", 100, 0.7),
          "rpart-100-0.7" = init("rpart", 100, 0.7),
          "lda-100-0.7" = init("lda", 100, 0.7),
          "ada-100-0.7" = init("ada", 100, 0.7)
  )
}

###############################################################################
# read the dataset into memory
read_dataset <- function() {

  # read training data-set (19622 x 160)
  f <- read.csv('data/pml-training.csv')
  if (DATASET_SIZE == 0) {
    DATASET_SIZE <- dim(f)[1]
  }
  
  # possibly take a sample for development. Later replace with all the input dataset
  dataset <- sample_n(f, size=DATASET_SIZE, replace=FALSE)
  
  # remove special rows from dataset
  dataset <- dataset[dataset$new_window != "yes", ]
  
  invisible(dataset)
}

###############################################################################
# read the quiz dataset into memory
read_quiz_dataset <- function() {
  dataset <- read.csv('data/pml-testing.csv')
  invisible(dataset)
}

###############################################################################
# extract columns from a dataset to use as features
extractFeatures <- function(ds) {
  # extract feature names
  n <- names(ds)
  inx <- grepl("^roll_|^pitch_|^yaw_|^total_|^gyros_|^accel_|^magnet_", tolower(n))
  
  # retuen the features
  invisible(n[inx])
}

###############################################################################
# returns a list with training_class, training_features, testing_class, testing_features dataframes
# to be used in an ML algorithm
get_ml_datasets <- function(dataset) {
  
  # create training and testing parttitions
  inx <- createDataPartition(dataset$classe, p=TRAINING_SIZE, list=FALSE) 
  
  # create a training and testing datasets. 
  training <- dataset[ inx, ]
  testing <- dataset[-inx, ]
  
  # get the features to use
  features <- extractFeatures(training)
  
  # separate features and classes for training data set
  training_features <- training[, features]
  training_classes <- as.factor(training$classe)
  
  # separate features and classes for testing data set
  testing_features <- testing[, features]
  testing_classes <- as.factor(testing$classe)
  
  # return multiple objects
  invisible(list(training_features = training_features,
                 training_classes = training_classes,
                 testing_features = testing_features,
                 testing_classes = testing_classes))
}

###############################################################################
# reads the training dataset and returns a list with training_class, training_features, 
# testing_class, testing_features dataframes to be used un a ML algorithm
read_ml_datasets <- function() {
  flog.info("Read datasets")

  dataset <- read_dataset()
  ds <- get_ml_datasets(dataset)
  
  flog.info("  training has %s rows", nrow(ds$training_features))
  flog.info("  testing has %s rows", nrow(ds$testing_features))
  
  ds
}

###############################################################################
# print a message with elapsed time since start_time
print_elapsed_time <- function(start_time, message) {
  cat(message, ':\n', sep = "")
  print(proc.time() - start_time)
}


###############################################################################
# save model
save_model <- function(model) {
  flog.info(paste0("Saving model ", MODEL_NAME))
  saveRDS(model, paste0("models/", MODEL_NAME, ".rds"))
}

###############################################################################
# reload model
read_model <- function(model_name) {
  flog.info(paste0("Reading model ", model_name))
  readRDS(paste0("models/", model_name, ".rds"))
}

###############################################################################
# print model results to console
print_model_results <- function(model, training_classes, testing_classes, testing_features) {
  
  # print put training results
  flog.info("Training results:")
  cm <- confusionMatrix(training_classes, predict(model))
  trainingAccuracy <- cm$overall[1]
  print(cm)
  cat("\n")
  
  # test model
  flog.info("Testing results:")
  testing_predic <- predict(model, newdata = testing_features)
  accuracy <- 0.0
  if (!is.null(testing_classes)) {
    # we're developing model and know the testing classes
    cm <- confusionMatrix(testing_classes, testing_predic)
    testingAccuracy <- cm$overall[1]
    print(cm)
    cat("\n")
  }
  
  # print model
  flog.info("Model obtained:")
  print(model)
  cat("\n")
  
  invisible(list(trainingAccuracy = trainingAccuracy, 
                 testingAccuracy = testingAccuracy))
}

###############################################################################
# run ML analysis
#
# method = rf | rpart | lda | ada
# size = size of the dataset or 0 for all
# training_size = percentage of data to use for training
# train_control = optional. Training control parameters
run_analysis <- function(method, size, training_size, train_control = trainControl(), post_name = "") {
  
  # track total execution time
  ptm <- proc.time()
  
  # set logging level
  flog.threshold(INFO)
  
  flog.info("Start analysis")
  
  # set model and parameters to use
  init(method, size, training_size, post_name)
  
  # read datasets
  ds <- read_ml_datasets()
  
  #print(str(ds$training_classes))
  
  # train model
  flog.info(paste0("Start training ", MODEL_NAME))
  model1 <- train(ds$training_classes ~ ., method = METHOD, data = ds$training_features, trControl = train_control)
  flog.info(paste0("Finished training ", MODEL_NAME))
  
  # print all model results to log
  acc <- print_model_results(model1, ds$training_class, ds$testing_class, ds$testing_features)
  
  # save model
  save_model(model1)
  
  # final accuracy for testing dataset
  cat("\n")
  flog.info(paste0("Training accuracy = ", as.character(acc$trainingAccuracy)))
  flog.info(paste0("Testing  accuracy = ", as.character(acc$testingAccuracy)))
  cat("\n")
  
  flog.info("End analysis")
  print(proc.time() - ptm)
}

###############################################################################
# run PCA analysis
#
# training_size = percentage of data to use for training
# tresh (default = 90%) for pca analysis
run_pca_analysis <- function(training_size, thresh = 0.90) {
  
  # track total execution time
  ptm <- proc.time()
  
  # set logging level
  flog.threshold(INFO)
  
  flog.info("Start analysis")
  
  # set model and parameters to use
  init_defaults()
  
  # read datasets
  ds <- read_ml_datasets()
  
  #print(str(ds$training_classes))
  
  # train model
  flog.info(paste0("Start PCA "))
  preProc <- preProcess(ds$training_features, method="pca", thresh)   #,pcaComp=2)
  print(preProc)
  flog.info(paste0("Finished PCA "))
  
  flog.info("End analysis")
  print(proc.time() - ptm)
}

###############################################################################
# Plots results from a saved model
# method = rf | rpart | lda | ada
# size = size of the dataset or 0 for all
# training_size = percentage of data to use for training
# post_name = string to add to MODEL_NAME at the end (ex. trControl parameter info)
# 
# will store arguments in global variables METHOD, DATASET_SIZE, TRAINING_SIZE, 
# MODEL_NAME and set the seed for reproducibility
plot_model <- function(method, size, training_size, post_name = "") {
  
  # specify model to analyze
  init(method, size, training_size, post_name)
  
  # read model
  model1 <- read_model(MODEL_NAME)
  
  # print model
  flog.info("Model obtained:")
  print(model1)
  cat("\n")
  
  # print model
  flog.info("Final model obtained:")
  print(model1$finalModel)
  cat("\n")
  
  # plotting resampling profile
  # trellis.par.set(caretTheme())
  
  #plot(model1)
  
  # plot the random forest model
  m <- model1$finalModel
  layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
  par(mar=c(5,4,4,0)) #No margin on the right side
  plot(m, log="y", main="Errors by number of trees")
  par(mar=c(5,0,4,2)) #No margin on the left side
  plot(c(0,1),type="n", axes=F, xlab="", ylab="")
  legend("top", colnames(m$err.rate),col=1:4,cex=.8,fill=1:4,box.lty=0)
}

