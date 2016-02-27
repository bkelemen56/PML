# source common variables and functions
source('common.R')

# read model
model1 <- read_model("newmodel-rf-0.8-cv-10")

# read quiz dataset
quiz_testing_ds <- read_quiz_dataset()

flog.info("Testing results:")
answers <- predict(model1, newdata = quiz_testing_ds)

# print answers
print(answers)

# print easier to read
print(t(t(answers)))
