# source common variables and functions
source('common.R')

# function to create scatter plots
scatterPlot <- function(pattern) {
  featurePlot(x = sample_features[, grep(pattern, names(sample_features), value = TRUE)],
              y = sample_classes,
              plot = "pairs",
              main = paste0("Feature Plot ", pattern),
              ## Add a key at the top
              auto.key = list(columns = 5))  
}

# read datasets
ds <- read_ml_datasets()

# perform some visualizations on a sample of the data
sample_idx <- sample(length(ds$training_classes), 500)
sample_features <- ds$training_features[sample_idx, ]
sample_classes <- ds$training_classes[sample_idx]

# first plots: nothing earthing
scatterPlot('^roll_')
scatterPlot('^pitch_')
scatterPlot('^yaw_')
scatterPlot('^total_')

scatterPlot('^gyros_belt')
scatterPlot('^gyros_arm')
scatterPlot('^gyros_forearm')
scatterPlot('^gyros_dumbbell')

scatterPlot('^accel_belt')
scatterPlot('^accel_arm')
scatterPlot('^accel_forearm')
scatterPlot('^accel_dumbbell')

scatterPlot('^magnet_belt')
scatterPlot('^magnet_arm')
scatterPlot('^magnet_forearm')
scatterPlot('^magnet_dumbbell')

scatterPlot('_belt*')
scatterPlot('_arm*')
scatterPlot('_forearm*')
scatterPlot('_dumbbell*')
