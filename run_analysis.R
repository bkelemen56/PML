# source common variables and functions
source('common.R')

# run the ML analysis
#run_analysis("rf", 1000, 0.7)
#run_analysis("lda", 0, 0.7)
#run_analysis("rpart", 0, 0.7)
#run_analysis("ada", 1000, 0.7)   # fails
#run_analysis("nb", 0, 0.7)

# rf with trainControl paramenters
#run_analysis("rf", 0, 0.7, train_control = trainControl(method="cv", number=10), post_name = "cv-10")
#run_analysis("rf", 0, 0.7, train_control = trainControl(method="repeatedcv", number=10, repeats=3), post_name = "repeatedcv-10-3")
#run_analysis("rf", 1000, 0.7, train_control = trainControl(method="boot", number=100), post_name = "boot-10")
#run_analysis("rf", 0, 0.7, train_control = trainControl(method="boot", number=25), post_name = "boot-25")
#run_analysis("rf", 0, 0.7, train_control = trainControl(method="LOOCV"), post_name = "LOOCV")  ##  DIDN'T FINISH

# Final runs 20160216
# run the ML analysis
# run_analysis("rf", 0, 0.8)
# run_analysis("lda", 0, 0.8)
# run_analysis("rpart", 0, 0.8)
# run_analysis("nb", 0, 0.8)
# run_analysis("svmLinear", 0, 0.8) 

run_analysis("rf", 0, 0.8, train_control = trainControl(method="cv", number=10), post_name = "cv-10")
#run_analysis("rf", 0, 0.8, train_control = trainControl(method="repeatedcv", number=10, repeats=3), post_name = "repeatedcv-10-3")
#run_analysis("rf", 0, 0.8, train_control = trainControl(method="boot", number=25), post_name = "boot-25")

#run_analysis("svmLinear", 0, 0.8, train_control = trainControl(method="cv", number=10), post_name = "cv-10") 
#run_analysis("svmLinear", 0, 0.8, train_control = trainControl(method="repeatedcv", number=10, repeats=3), post_name = "repeatedcv-10-3") 

#run_analysis("lda", 0, 0.8, train_control = trainControl(method="cv", number=10), post_name = "cv-10")
#run_analysis("lda", 0, 0.8, train_control = trainControl(method="repeatedcv", number=10, repeats=3), post_name = "repeatedcv-10-3")

#run_analysis("rpart", 0, 0.8, train_control = trainControl(method="cv", number=10), post_name = "cv-10")
#run_analysis("rpart", 0, 0.8, train_control = trainControl(method="repeatedcv", number=10, repeats=3), post_name = "repeatedcv-10-3")

#run_analysis("nb", 0, 0.8, train_control = trainControl(method="cv", number=10), post_name = "cv-10")
#run_analysis("nb", 0, 0.8, train_control = trainControl(method="repeatedcv", number=10, repeats=3), post_name = "repeatedcv-10-3")

# Don't run - very slow...
#run_analysis("AdaBag", 0, 0.8)    # very slow
#run_analysis("AdaBag", 100, 0.8, train_control = trainControl(method="cv", number=10), post_name = "cv-10") 
#run_analysis("AdaBoost.M1", 100, 0.8) 

# PCA analysis
#run_pca_analysis(0, 0.9)  # 16 pca explain 90% of variance
#run_pca_analysis(0, 0.95)  # 20 pca explain 95% of variance



