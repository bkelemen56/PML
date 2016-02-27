library(caret)

# reload model
model <- readRDS("model-1000-rpart.rds")

# plot tree
plot(model$finalModel, uniform=TRUE, main="Classification Tree")
text(model$finalModel, use.n=TRUE, all=TRUE, cex=.8)

# fancy plot (doesn't work)
library(rattle)
fancyRpartPlot(model$finalModel)