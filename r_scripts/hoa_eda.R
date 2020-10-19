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

# modelling
library('recipes')
library('rsample')
library('keras')
library('tfdatasets')

# function to extract binomial confidence levels
get_binCI <- function(x,n) as.list(setNames(binom.test(x,n)$conf.int, c("lwr", "upr")))

path <- paste(getwd(), '/../input/', sep = "")
train <- vroom(str_c(path,'train_features.csv'), col_types = cols())
targets <- vroom(str_c(path, "train_targets_scored.csv"), col_types = cols())
targets_non <- vroom(str_c(path, "train_targets_nonscored.csv"), col_types = cols())
test <- vroom(str_c(path,'test_features.csv'), col_types = cols())
sample_submit <- vroom(str_c(path,'sample_submission.csv'), col_types = cols())

### 3 Overview: File structure and content
head(train, 50) %>% 
  DT::datatable()

test %>% 
  head() %>% 
  DT::datatable()

targets %>% 
  head(50) %>% 
  DT::datatable()

targets_non %>% 
  head(50) %>% 
  DT::datatable()

sum(is.na(train))

sum(is.na(test))

non_zero <- targets %>% 
  select(-sig_id) %>% 
  na_if(1) %>% 
  is.na()

non_zero_percent <-  sum(non_zero) / (nrow(non_zero) * ncol(non_zero)) * 100

sprintf("Percentage of non-zero target class values: %.3f%%", non_zero_percent)

non_zero <- targets_non %>% 
  select(-sig_id) %>% 
  na_if(1) %>% 
  is.na()

non_zero_percent <-  sum(non_zero) / (nrow(non_zero) * ncol(non_zero)) * 100

sprintf("Percentage of non-zero non-scored target class values: %.3f%%", non_zero_percent)

nrow(train) - nrow(train %>% distinct(sig_id))

nrow(targets) - nrow(targets %>% distinct(sig_id))

train %>% 
  select(sig_id) %>% 
  anti_join(targets, by = "sig_id") %>% 
  nrow()

### 4 Individual feature visualisations
p1 <- train %>% 
  count(cp_type) %>% 
  add_tally(n, name = "total") %>% 
  mutate(perc = n/total) %>% 
  ggplot(aes(cp_type, perc, fill = cp_type)) +
  geom_col() +
  geom_text(aes(label = sprintf("%s", n)), nudge_y = 0.02) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("grey70", "violetred")) +
  theme_hc() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", fill = "State", title = "Sample treatment", subtitle = "(Compound vs Control)")

p2 <- train %>% 
  count(cp_dose) %>% 
  add_tally(n, name = "total") %>% 
  mutate(perc = n/total) %>% 
  ggplot(aes(cp_dose, perc, fill = cp_dose)) +
  geom_col() +
  geom_text(aes(label = sprintf("%s", n)), nudge_y = 0.02) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("darkblue", "darkred")) +
  theme_hc() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", fill = "State", title = "Treatment Dose", subtitle = "(high vs low)")

p3 <- train %>% 
  count(cp_time) %>% 
  mutate(cp_time = as.factor(cp_time)) %>% 
  add_tally(n, name = "total") %>% 
  mutate(perc = n/total) %>% 
  ggplot(aes(cp_time, perc, fill = cp_time)) +
  geom_col() +
  geom_text(aes(label = sprintf("%s", n)), nudge_y = 0.01) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(type = "seq", palette = "Oranges") +
  theme_hc() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", fill = "State", title = "Treatment duration", subtitle = "(Units of hours)")

p1 + p2 + p3

# 4.2
train %>% 
  select(sig_id, starts_with("g-")) %>% 
  select(seq(1,5)) %>% 
  pivot_longer(starts_with("g-"), names_to = "feature", values_to = "value") %>% 
  ggplot(aes(value, fill = feature)) +
  geom_density() +
  facet_wrap(~ feature) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(x = "", y = "", fill = "State", title = "Distributions for gene expression features")

gene_stats <- train %>% 
  select(starts_with("g-")) %>% 
  summarise(across(everything(), list(min = min, max = max, mean = mean, sd = sd))) %>% 
  pivot_longer(starts_with("g-"), names_to = "features", values_to = "values") %>% 
  separate(features, into = c("features", "stat"), sep = "_")

gene_stats %>% 
  ggplot(aes(values, fill = stat)) +
  geom_density() +
  scale_fill_manual(values = wes_palette("GrandBudapest2")) +
  facet_wrap(~ stat, scales = "free") +
  theme_tufte() +
  theme(legend.position = "none") +
  labs(x = "", y = "", fill = "State", title = "Gene distribution meta statistics")

# 4.3
train %>% 
  select(sig_id, starts_with("c-")) %>% 
  select(seq(1,5)) %>% 
  pivot_longer(starts_with("c-"), names_to = "feature", values_to = "value") %>% 
  ggplot(aes(value, fill = feature)) +
  geom_density() +
  scale_fill_brewer(palette = "Set3") +
  facet_wrap(~ feature) +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", fill = "State", title = "Distributions for cell viability features")

train %>% 
  select(sig_id, starts_with("c-")) %>% 
  select(seq(1,7)) %>% 
  pivot_longer(starts_with("c-"), names_to = "feature", values_to = "value") %>% 
  filter(value < -4) %>% 
  ggplot(aes(value, fill = feature)) +
  geom_density() +
  scale_fill_brewer(palette = "Set3") +
  facet_wrap(~ feature) +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Cell viability features - zoom in on negative tail")

cell_stats <- train %>% 
  select(starts_with("c-")) %>% 
  summarise(across(everything(), list(min = min, max = max, mean = mean, sd = sd))) %>% 
  pivot_longer(starts_with("c-"), names_to = "features", values_to = "values") %>% 
  separate(features, into = c("features", "stat"), sep = "_")

cell_stats %>% 
  ggplot(aes(values, fill = stat)) +
  geom_density() +
  scale_fill_manual(values = wes_palette("GrandBudapest1")) +
  facet_wrap(~ stat, scales = "free") +
  theme_tufte() +
  theme(legend.position = "none") +
  labs(x = "", y = "", fill = "State", title = "Cell distribution meta statistics")

# 4.4
rowstats <- targets %>% 
  select(-sig_id) %>% 
  rowwise() %>% 
  mutate(sum = sum(c_across(everything()))) %>% 
  select(sum) %>% 
  ungroup()

rowstats %>% 
  count(sum) %>% 
  add_tally(n, name = "total") %>% 
  mutate(perc = n/total) %>% 
  mutate(sum = as.factor(sum)) %>% 
  ggplot(aes(sum, n, fill = sum)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.2f%%", perc*100)), nudge_y = 500) +
  # scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Number of Activations per Sample")

target_sums <- targets %>% 
  select(-sig_id) %>% 
  summarise(across(everything(), sum)) %>% 
  pivot_longer(everything(), names_to = "target", values_to = "sum")

p1 <- target_sums %>% 
  ggplot(aes(sum)) +
  geom_density(fill = "darkorange") +
  geom_vline(xintercept = 40, linetype = 2) +
  scale_x_log10() +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "MoA count per target class", subtitle = "Dashed line: 40")

p2 <- target_sums %>% 
  arrange(desc(sum)) %>% 
  head(5) %>% 
  mutate(target = str_replace_all(target, "_", " ")) %>% 
  ggplot(aes(reorder(target, sum, FUN = min), sum, fill = sum)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "blue1", high = "blue4") +
  scale_x_discrete(labels = function(x) lapply(str_wrap(x, width = 25), paste, collapse="\n")) +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Classes with most MoAs")

p3 <- target_sums %>% 
  arrange(sum) %>% 
  head(5) %>%  
  mutate(target = str_replace_all(target, "_", " ")) %>% 
  ggplot(aes(reorder(target, sum, FUN = min), sum, fill = sum)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "red4", high = "red1") +
  scale_x_discrete(labels = function(x) lapply(str_wrap(x, width = 25), paste, collapse="\n")) +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Classes with fewest MoAs")

p1 + (p2/p3)

target_sums %>% 
  separate(target, into = c("a", "b", "c", "d", "e", "type"), fill = "left") %>% 
  count(type) %>% 
  add_tally(n, name = "total") %>% 
  mutate(perc = n/total) %>% 
  filter(n > 1) %>% 
  ggplot(aes(reorder(type, n, FUN = min), n, fill = n)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.2f%%", perc*100)), nudge_y = 6) +
  coord_flip() +
  scale_fill_viridis() +
  scale_x_discrete(labels = function(x) lapply(str_wrap(x, width = 25), paste, collapse="\n")) +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Common final terms in class names")

### 5 Multiple feature interaction visuals
# 5.1
train %>% 
  group_by(cp_type, cp_dose, cp_time) %>% 
  count() %>% 
  mutate(cp_time = as.factor(cp_time)) %>% 
  ggplot(aes(cp_time, n, fill = cp_time)) +
  geom_col() +
  facet_grid(cp_dose ~ cp_type) +
  scale_fill_manual(values = wes_palette("IsleofDogs1")) +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "Treatment Duration", y = "", fill = "State",
       title = "Treatment Feature Interactions", subtitle = "Horizontal: type, Vertical: dose, Bars/Colour: duration")

train %>% 
  select(starts_with("g-")) %>% 
  select(seq(1,200)) %>% 
  cor(use="complete.obs", method = "pearson") %>%
  corrplot(type="lower", tl.col = "black",  diag=FALSE, method = "color",
           outline = FALSE, tl.pos = "n", cl.ratio = 0.05)

train %>% 
  select(starts_with("g-")) %>% 
  select(seq(1,20)) %>% 
  cor(use="complete.obs", method = "pearson") %>%
  corrplot(type="lower", tl.col = "black",  diag=FALSE, method = "color",
           cl.ratio = 0.1)

p1 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(g_0, g_8)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "darkred") +
  theme_minimal() +
  labs(title = str_c("g-0 vs g-8: coef = ", sprintf("%.2f", cor(train$`g-0`, train$`g-8`))))

p2 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(g_10, g_17)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "darkblue") +
  theme_minimal() +
  labs(title = str_c("g-10 vs g-17: coef = ", sprintf("%.2f", cor(train$`g-10`, train$`g-17`))))

p3 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(g_0, g_3)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "black") +
  theme_minimal() +
  labs(title = str_c("g-0 vs g-3: coef = ", sprintf("%.2f", cor(train$`g-0`, train$`g-3`))))

p4 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(g_10, g_19)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "black") +
  theme_minimal() +
  labs(title = str_c("g-10 vs g-19: coef = ", sprintf("%.2f", cor(train$`g-10`, train$`g-19`))))


(p1 + p2) / (p3 + p4)

train %>% 
  select(starts_with("c-")) %>% 
  cor(use="complete.obs", method = "pearson") %>%
  corrplot(type="lower", tl.col = "black",  diag=FALSE, method = "color",
           outline = FALSE, tl.pos = "n", cl.ratio = 0.05)

train %>% 
  select(starts_with("c-")) %>% 
  select(seq(1,10)) %>% 
  cor(use="complete.obs", method = "pearson") %>%
  corrplot(type="lower", tl.col = "black",  diag=FALSE, method = "number",
           cl.ratio = 0.1)

p1 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(c_1, c_2)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "darkblue") +
  theme_minimal() +
  labs(title = str_c("c-1 vs c-2: coef = ", sprintf("%.2f", cor(train$`c-1`, train$`c-2`))))

p2 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(c_3, c_4)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "darkblue") +
  theme_minimal() +
  labs(title = str_c("c-3 vs c-4: coef = ", sprintf("%.2f", cor(train$`c-3`, train$`c-4`))))

p3 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(c_5, c_9)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "darkblue") +
  theme_minimal() +
  labs(title = str_c("c-5 vs c-9: coef = ", sprintf("%.2f", cor(train$`c-5`, train$`c-9`))))

p4 <- train %>% 
  janitor::clean_names() %>% 
  ggplot(aes(c_0, c_6)) +
  geom_point(col = "grey40", size = 0.5) +
  geom_smooth(method = "lm", formula = "y~x", col = "darkblue") +
  theme_minimal() +
  labs(title = str_c("c-0 vs c-6: coef = ", sprintf("%.2f", cor(train$`c-0`, train$`c-6`))))

(p1 + p2) / (p3 + p4)

# 5.2 Interactions between sets of features
train %>% 
  janitor::clean_names() %>% 
  select(cp_dose, cp_time, g_525, g_666, c_42, c_22) %>% 
  mutate(cp_time = as.factor(str_c("Duration ", cp_time, "h"))) %>% 
  pivot_longer(starts_with(c("g_", "c_")), names_to = "feature", values_to = "value") %>% 
  ggplot(aes(value, fill = cp_dose)) +
  geom_density(alpha = 0.5) +
  facet_grid(feature ~ cp_time) +
  theme_minimal() +
  theme(legend.position = "top") +
  labs(x = "Feature value", fill = "Dose", title = "Treatment features vs example cell & gene distributions")

train %>% 
  select(cp_time, cp_dose, starts_with("c-")) %>% 
  mutate(cp_time = as.factor(str_c("Duration ", cp_time, "h"))) %>% 
  select(seq(1,5)) %>% 
  pivot_longer(starts_with("c-"), names_to = "feature", values_to = "value") %>% 
  filter(value < -4) %>% 
  ggplot(aes(value, fill = cp_dose)) +
  geom_density(alpha = 0.5) +
  facet_grid(feature ~ cp_time) +
  theme_minimal() +
  theme(legend.position = "top", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", fill = "Dose", title = "Treatment vs cell - zoom in on negative tail")

train %>% 
  janitor::clean_names() %>% 
  select(cp_type, g_8, g_525, g_666, c_14, c_42, c_22) %>% 
  pivot_longer(starts_with(c("g_", "c_")), names_to = "feature", values_to = "value") %>% 
  ggplot(aes(value, fill = cp_type)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ feature)+
  theme_minimal() +
  theme(legend.position = "top") +
  labs(x = "Feature value", fill = "Type", title = "Treatment features vs example cell & gene distributions")

stats_all <- train %>% 
  select(starts_with("cp"), num_range(prefix = "g-", c(8, 525)), num_range(prefix = "c-", c(14, 42))) %>% 
  bind_cols(rowstats)

stats_all %>% 
  group_by(cp_type, sum) %>% 
  summarise(n = n()) %>% 
  add_tally(n, name = "total") %>%
  ungroup() %>% 
  mutate(perc = n/total) %>% 
  mutate(sum = as.factor(sum)) %>% 
  ggplot(aes(sum, n, fill = sum)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.2f%%", perc*100)), nudge_y = 500) +
  scale_fill_brewer(palette = "Set2") +
  facet_grid(~ cp_type, scales = "free_x", space = "free_x") +
  theme_hc() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Number of Activations per type")

stats_all %>% 
  filter(cp_type == "trt_cp") %>% 
  mutate(cp_time = as.factor(str_c("Duration ", cp_time, "h"))) %>% 
  mutate(sum = if_else(sum >= 3, "3+", as.character(sum))) %>% 
  mutate(sum = as.factor(sum)) %>% 
  pivot_longer(starts_with(c("g-", "c-")), names_to = "feature", values_to = "value") %>% 
  ggplot(aes(sum, value, fill = sum)) +
  # geom_violin(draw_quantiles = c(0.25, 0.75)) +
  geom_violin() +
  facet_grid(feature ~ cp_time) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(x = "Sum of active MoAs per row", y = "Cell or Gene values", fill = "Rowwise sum of MoAs",
       title = "Selected cell & gene distributions for different counts of MoAs per row",
       subtitle = "Facetted by cell/gene vs treatment duration")

stats_all %>% 
  filter(cp_type == "trt_cp") %>% 
  mutate(sum = if_else(sum >= 3, "3+", as.character(sum))) %>% 
  mutate(sum = as.factor(sum)) %>% 
  pivot_longer(starts_with(c("g-", "c-")), names_to = "feature", values_to = "value") %>% 
  ggplot(aes(value, fill = cp_dose)) +
  geom_density(alpha = 0.5) +
  facet_grid(feature ~ sum) +
  theme_minimal() +
  theme(legend.position = "top") +
  labs(y = "", x = "Cell or Gene values", fill = "Dose",
       title = "Selected cell & gene distributions for different counts of MoAs per row",
       subtitle = "Colour-coded treatment dose")

foo <- train %>% 
  select(starts_with("cp"), num_range(prefix = "g-", c(8, 525)), num_range(prefix = "c-", c(14, 42))) %>% 
  bind_cols(rowstats) %>% 
  bind_cols(targets %>% select(dopamine_receptor_antagonist)) %>% 
  filter(cp_type == "trt_cp") %>% 
  pivot_longer(starts_with(c("g-", "c-")), names_to = "feature", values_to = "value") %>% 
  select(-cp_type) %>% 
  mutate(dopamine_receptor_antagonist = as.factor(dopamine_receptor_antagonist))

foo %>% 
  ggplot(aes(value, fill = dopamine_receptor_antagonist)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ feature) +
  theme_minimal() +
  theme(legend.position = "top") +
  labs(y = "", x = "Cell or Gene values", fill = "Class value",
       title = "Selected cell & gene distributions for specific class",
       subtitle = "Example: dopamine receptor antagonist")

foo <- train %>% 
  select(cp_type, num_range(prefix = "c-", seq(1,4))) %>% 
  bind_cols(rowstats) %>% 
  bind_cols(targets %>% select(dopamine_receptor_antagonist, cyclooxygenase_inhibitor)) %>% 
  filter(cp_type == "trt_cp") %>% 
  pivot_longer(starts_with(c("g-", "c-")), names_to = "feature", values_to = "value") %>% 
  select(-cp_type) %>% 
  filter(value < -1) %>% 
  pivot_longer(c(dopamine_receptor_antagonist, cyclooxygenase_inhibitor), names_to = "class_name", values_to = "class_value") %>% 
  mutate(class_value = as.factor(class_value))

foo %>% 
  ggplot(aes(value, fill = class_value)) +
  geom_density(alpha = 0.5) +
  facet_grid(class_name ~ feature) +
  theme_minimal() +
  theme(legend.position = "top") +
  labs(y = "", x = "Cell values", fill = "Class value",
       title = "First cell feature distributions for specific class",
       subtitle = "Example: dopamine receptor antagonist")

### 6 Non-scored targets
rowstats_non <- targets_non %>% 
  select(-sig_id) %>% 
  rowwise() %>% 
  mutate(sum = sum(c_across(everything()))) %>% 
  select(sum) %>% 
  ungroup()

target_sums_non <- targets_non %>% 
  select(-sig_id) %>% 
  summarise(across(everything(), sum)) %>% 
  pivot_longer(everything(), names_to = "target", values_to = "sum")

p1 <- rowstats_non %>% 
  count(sum) %>% 
  add_tally(n, name = "total") %>% 
  mutate(perc = n/total) %>% 
  mutate(sum = as.factor(sum)) %>% 
  ggplot(aes(sum, n, fill = sum)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.2f%%", perc*100)), nudge_y = 1000) +
  # scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Set2") +
  theme_tufte() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Number of Activations per Sample Row")

p2 <- target_sums_non %>% 
  ggplot(aes(sum)) +
  geom_density(fill = "darkorange") +
  geom_vline(xintercept = 6, linetype = 2) +
  scale_x_continuous(trans = "log1p", breaks = c(0, 10, 20, 50, 100)) +
  theme_tufte() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "MoA count per target class", subtitle = "Dashed line: 6")

p3 <- target_sums_non %>% 
  arrange(desc(sum)) %>% 
  head(5) %>% 
  mutate(target = str_replace_all(target, "_", " ")) %>% 
  ggplot(aes(reorder(target, sum, FUN = min), sum, fill = sum)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "blue1", high = "blue4") +
  scale_x_discrete(labels = function(x) lapply(str_wrap(x, width = 25), paste, collapse="\n")) +
  theme_tufte() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Classes with most MoAs")

p1 / (p2 + p3) + plot_annotation(title = 'Non-scored target data')

target_sums_non %>% 
  separate(target, into = c("a", "b", "c", "d", "e", "f", "g", "type"), fill = "left") %>% 
  count(type) %>% 
  add_tally(n, name = "total") %>% 
  mutate(perc = n/total) %>% 
  filter(n > 1) %>% 
  ggplot(aes(reorder(type, n, FUN = min), n, fill = n)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.2f%%", perc*100)), nudge_y = 12) +
  coord_flip() +
  scale_fill_viridis(option = "cividis") +
  scale_x_discrete(labels = function(x) lapply(str_wrap(x, width = 25), paste, collapse="\n")) +
  theme_minimal() +
  theme(legend.position = "none", plot.subtitle = element_text(size = 10)) +
  labs(x = "", y = "", title = "Non-scored targets: Common final terms in class names")

### 7 Dimensionality reduction via PCA
# 7.1
X <- train %>% 
  select(starts_with("g-"))

pca <- prcomp(X, center = TRUE, scale. = TRUE)

p1 <- fviz_eig(pca, title = "Variance", ncp = 5)
p2 <- fviz_pca_var(pca,
                   col.var = "contrib",
                   gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                   repel = TRUE,
                   label = "none",
                   title = "Variables"
)

p1 + p2 + plot_annotation(title = 'PCA on gene features: overview')

p1 <- fviz_contrib(pca, choice = "var", axes = 1, top = 15)
p2 <- fviz_contrib(pca, choice = "var", axes = 2, top = 15)

p1 / p2 + plot_annotation(title = 'PCA on gene features: variable contributions')

p1 <- fviz_pca_ind(pca, label = "none",
                   habillage = train %>% mutate(cp_type = if_else(cp_type == "trt_cp", "Treatment compound", "Treatment control")) %>% pull(cp_type),
                   # habillage = train$cp_type,
                   alpha.ind = 0.3,
                   palette = c("#FFCC00", "black"),
                   title = "Treatment type") +
  theme(legend.position = "top")

p2 <- fviz_pca_ind(pca, label = "none",
                   habillage = train$cp_dose,
                   alpha.ind = 0.5,
                   palette = wes_palette("Cavalcanti1"),
                   title = "Treatment dose") +
  theme(legend.position = "top")

p3 <- fviz_pca_ind(pca, label = "none",
                   habillage = train$cp_time,
                   alpha.ind = 1,
                   palette = "Accent",
                   title = "Treatment duration") +
  theme(legend.position = "top")


p4 <- fviz_pca_ind(pca, label = "none",
                   habillage = rowstats %>% mutate(sum = if_else(sum >= 3, "3+", as.character(sum))) %>% pull(sum),
                   palette = "Dark2",
                   alpha.ind = 0.7,
                   title = "Sum of MoAs") +
  theme(legend.position = "top")

(p1 + p2) / (p3 + p4) + plot_annotation(title = 'PCA on gene features by feature groups')

# 7.2
Xc <- train %>% 
  select(starts_with("c-"))

pca_cell <- prcomp(Xc, center = TRUE, scale. = TRUE)

p1 <- fviz_eig(pca_cell, title = "Variance", ncp = 5)
p2 <- fviz_pca_var(pca_cell,
                   col.var = "contrib",
                   gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                   repel = TRUE,
                   label = "none",
                   title = "Variables"
)
p3 <- fviz_contrib(pca_cell, choice = "var", axes = 1, top = 8)
p4 <- fviz_contrib(pca_cell, choice = "var", axes = 2, top = 8)

(p1 + p2) / (p3 + p4) + plot_annotation(title = 'PCA on cell features')

p1 <- fviz_pca_ind(pca_cell, label = "none",
                   habillage = train %>% mutate(cp_type = if_else(cp_type == "trt_cp", "Treatment compound", "Treatment control")) %>% pull(cp_type),
                   # habillage = train$cp_type,
                   alpha.ind = 0.3,
                   palette = c("#FFCC00", "black"),
                   title = "Treatment type") +
  theme(legend.position = "top")

p2 <- fviz_pca_ind(pca_cell, label = "none",
                   habillage = train$cp_dose,
                   alpha.ind = 0.5,
                   palette = wes_palette("Cavalcanti1"),
                   title = "Treatment dose") +
  theme(legend.position = "top")

p3 <- fviz_pca_ind(pca_cell, label = "none",
                   habillage = train$cp_time,
                   alpha.ind = 1,
                   palette = "Accent",
                   title = "Treatment duration") +
  theme(legend.position = "top")


p4 <- fviz_pca_ind(pca_cell, label = "none",
                   habillage = rowstats %>% mutate(sum = if_else(sum >= 3, "3+", as.character(sum))) %>% pull(sum),
                   palette = "Dark2",
                   alpha.ind = 0.7,
                   title = "Sum of MoAs") +
  theme(legend.position = "top")

(p1 + p2) / (p3 + p4) + plot_annotation(title = 'PCA on cell features by feature groups')

### 8 Baseline Model
# 8.1
training <- train %>%
  left_join(targets %>% rename_with(.fn = ~paste0("target_", .), .cols = -sig_id),
            by = "sig_id")

set.seed(4321)
tt_split <- initial_split(training, prop = 0.8, strata = cp_type)

X_train_pre <- training(tt_split) %>% 
  select(-starts_with("target"), -sig_id)
y_train <- training(tt_split) %>% 
  select(starts_with("target")) %>% 
  as.matrix()

X_valid_pre <- testing(tt_split) %>% 
  select(-starts_with("target"), -sig_id)
y_valid <- testing(tt_split) %>% 
  select(starts_with("target")) %>% 
  as.matrix()

rm(training)

cl_rec <- X_train_pre %>% 
  recipe() %>% 
  update_role(everything(), new_role = "predictor") %>% 
  step_integer(c(cp_type, cp_dose), zero_based = TRUE) %>%
  step_normalize(cp_time) %>%
  step_pca(starts_with("g-"), threshold = 0.95, prefix = "pcg_") %>%
  step_pca(starts_with("c-"), threshold = 0.95, prefix = "pcc_")

X_train <- cl_rec %>% prep() %>% juice() %>% as.matrix()
X_valid = cl_rec %>% prep() %>% bake(X_valid_pre) %>% as.matrix()
X_test = cl_rec %>% prep() %>% bake(test) %>% as.matrix()

glue("Number of columns reduced from { ncol(train) } to { ncol(X_train) }")

model <- keras_model_sequential() %>%
  layer_dense(units = 2048, activation = "relu", input_shape = ncol(X_train)) %>% 
  layer_dense(units = 1024, activation = "relu") %>% 
  layer_dropout(0.2) %>%
  layer_dense(units = ncol(y_train), activation = "sigmoid")

model

model %>% compile(
  optimizer = optimizer_adam(lr = 1e-4),
  loss = "binary_crossentropy"
)

cb_early_stopping <- callback_early_stopping(patience = 5, restore_best_weights = TRUE)

tic()
history <- model %>% fit(X_train, y_train,
                         epochs = 20,
                         verbose = 0,
                         batch_size = 32,
                         callbacks = list(cb_early_stopping),
                         validation_data = list(X_valid, y_valid)
)
toc()

#https://github.com/rstudio/keras/issues/1116

# set plot.keras_training_history in your global env to call user functions
plot.keras_training_history <- keras:::plot.keras_training_history
environment(plot.keras_training_history) <- globalenv()

# replace as.dataframe with custom function
as.data.frame.keras_training_history <- function (x, ...){
  
  if (tensorflow::tf_version() < "2.2") 
    x$metrics <- x$metrics[x$params$metrics]
  values <- x$metrics
  
  # pad <- x$params$epochs - length(values$loss)
  # pad_data <- list()
  # for (metric in x$params$metrics) pad_data[[metric]] <- rep_len(NA, 
  #                                                                pad)
  # values <- rbind(values, pad_data)
  
  values[] <- lapply(values, `length<-`, x$params$epochs)
  df <- data.frame(epoch = seq_len(x$params$epochs), value = unlist(values), 
                   metric = rep(sub("^val_", "", names(x$metrics)), each = x$params$epochs),
                   data = rep(grepl("^val_", names(x$metrics)), each = x$params$epochs))
  rownames(df) <- NULL
  df$data <- factor(df$data, c(FALSE, TRUE), c("training", "validation"))
  df$metric <- factor(df$metric, unique(sub("^val_", "", names(x$metrics))))
  df
  
}

plot(history) +
  coord_cartesian(xlim = c(0, 15), ylim = c(0.01, 0.02)) +
  theme_minimal() +
  labs(title = "Learning curves")

pred <- model$predict(X_test)
colnames(pred) <- str_remove_all(colnames(y_valid), "target_")
pred <- pred %>%
  as_tibble()

pred[test$cp_type == "ctl_vehicle",] <- 0

submit <- test %>%
  select(sig_id) %>%
  bind_cols(as_tibble(pred))

head(submit) %>% 
  DT::datatable()

c(identical(dim(submit), dim(sample_submit)),
  identical(submit$sig_id, sample_submit$sig_id),
  identical(colnames(submit), colnames(sample_submit)))

submit %>%
  write_csv("submission.csv")

