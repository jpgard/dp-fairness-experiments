library(ggplot2)
library(ggthemes)
library(magrittr)
library(gridExtra)
library(magrittr)

german_results = read.csv("/Users/jpgard/Documents/github/fair-robust/results.csv")
adult_results_sex = read.csv("/Users/jpgard/Documents/github/fair-robust/adult-sex-bs512e1000l1.0-sex-results.csv")
adult_results_race = read.csv("/Users/jpgard/Documents/github/fair-robust/adult-racebs512e1000l1.0-race-results.csv")
compas_results_race = read.csv("/Users/jpgard/Documents/github/fair-robust/compas-racebs64e1000l1.0-race-results.csv")

#+++++++++++++++++++++++++
# Function to calculate the mean and the standard deviation
# for each group
#+++++++++++++++++++++++++
# data : a data frame
# varname : the name of a column containing the variable
#to be summariezed
# groupnames : vector of column names to be used as
# grouping variables
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}


# Plot for german results
german_results$train_subset <- forcats::fct_recode(german_results$train_subset, "Union"="all", 
                    "Minority"="minority", "Majority"="majority")
german_results$eval_subset <- forcats::fct_recode(german_results$eval_subset, "Union"="all", 
                                                   "Minority"="minority", "Majority"="majority")
p_german <- german_results %>% 
  data_summary(varname="loss", 
               groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
                ) +
  ggtitle("German Credit Dataset\nSensitive Attribute: Sex\n(n = 1,000)") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
p_german

# Plot for adult results (sex)
adult_results_sex$train_subset <- forcats::fct_recode(adult_results_sex$train_subset, "Union"="all", 
                                                   "Minority"="minority", "Majority"="majority")
adult_results_sex$eval_subset <- forcats::fct_recode(adult_results_sex$eval_subset, "Union"="all", 
                                                  "Minority"="minority", "Majority"="majority")
p_adult_sex <- adult_results_sex %>%
  data_summary(varname="loss",
               groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("Adult Dataset\nSensitive Attribute: Sex\n(n = 37,000)") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
p_adult_sex

# Plot for adult results (race)
adult_results_race$train_subset <- forcats::fct_recode(adult_results_race$train_subset, "Union"="all", 
                                                      "Minority"="minority", "Majority"="majority")
adult_results_race$eval_subset <- forcats::fct_recode(adult_results_race$eval_subset, "Union"="all", 
                                                     "Minority"="minority", "Majority"="majority")
p_adult_race <- adult_results_race %>% data_summary(varname="loss", 
                              groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("Adult Dataset\nSensitive Attribute: Race\n(n = 37,000)") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
p_adult_race

# Plot for compas results (race)
compas_results_race$train_subset <- forcats::fct_recode(compas_results_race$train_subset, "Union"="all", 
                                                       "Minority"="minority", "Majority"="majority")
compas_results_race$eval_subset <- forcats::fct_recode(compas_results_race$eval_subset, "Union"="all", 
                                                      "Minority"="minority", "Majority"="majority")
p_compas_race <- compas_results_race %>% data_summary(varname="loss", 
                                    groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("COMPAS Dataset\nSensitive Attribute: Race\n(n = 5,200)") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) +
labs(caption = "Note: 'Minority' attribute group is larger than 'Majority' in COMPAS.")
p_compas_race

grid.arrange(p_german, p_compas_race, p_adult_sex, p_adult_race, ncol=4)
library(ggpubr)
ggarrange(p_german, p_compas_race, p_adult_sex, p_adult_race, nrow=1, common.legend = TRUE, legend="bottom")
ggsave("./subset_experiments_fairness_datasets.pdf", device="pdf", width=16, height=7)
