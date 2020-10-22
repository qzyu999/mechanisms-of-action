### 2 Preparations
# general visualisation
library('ggplot2') # visualisation
library('scales') # visualisation
library('patchwork') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation
library('ggthemes') # visualisation
library('viridis') # visualisation
library('glue') # visualisation

# general data manipulation
library('dplyr') # data manipulation
library('readr') # input/output
library('vroom') # input/output
library('skimr') # overview
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('janitor') # data cleaning
library('tictoc') # timing

# specific visualisation
library('alluvial') # visualisation
library('ggrepel') # visualisation
library('ggforce') # visualisation
library('ggridges') # visualisation
library('gganimate') # animations
library('GGally') # visualisation
library('wesanderson') # visualisation

# specific data manipulation
library('lazyeval') # data wrangling
library('broom') # data wrangling
library('purrr') # data wrangling
library('reshape2') # data wrangling
library('rlang') # encoding

# dimensionality reduction
library('factoextra')

# function to extract binomial confidence levels
get_binCI <- function(x,n) as.list(setNames(binom.test(x,n)$conf.int, c("lwr", "upr")))

path <- paste(getwd(), '/../input/', sep = "")
train <- vroom(str_c(path,'train_features.csv'), col_types = cols())
targets <- vroom(str_c(path, "train_targets_scored.csv"), col_types = cols())
targets_non <- vroom(str_c(path, "train_targets_nonscored.csv"), col_types = cols())
test <- vroom(str_c(path,'test_features.csv'), col_types = cols())

# Training set size
dim(train)
dim(test)
(nrow(test)*4) / nrow(train)

colnames(train)

dim(targets)
