library(ggplot2)
library(ggthemes)
library(magrittr)
library(gridExtra)
library(magrittr)

german_results = read.csv("/Users/jpgard/Documents/github/fair-robust/results.csv")
german_results_dp_S1z0.8 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/german-sexbs64lr0.01e1000l1.0dpTrueclip1.0z0.8-sex-results.csv")
german_results_dp_S10z0.4 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/german-sexbs64lr0.01e1000l1.0dpTrueclip10.0z0.4-sex-results.csv")

adult_results_sex = read.csv("/Users/jpgard/Documents/github/fair-robust/adult-sex-bs512e1000l1.0-sex-results.csv")
adult_results_sex_dp_S1z0.8 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/adult-sexbs512lr0.01e1000l1.0dpTrueclip1.0z0.8-sex-results.csv")
adult_results_sex_dp_S10z0.4 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/adult-sexbs512lr0.01e1000l1.0dpTrueclip10.0z0.4-sex-results.csv")

adult_results_race = read.csv("/Users/jpgard/Documents/github/fair-robust/adult-racebs512e1000l1.0-race-results.csv")
adult_results_race_dp_S1z0.8 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/adult-racebs512lr0.01e1000l1.0dpTrueclip1.0z0.8-race-results.csv")
adult_results_race_dp_S10z0.4 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/adult-racebs512lr0.01e1000l1.0dpTrueclip10.0z0.4-race-results.csv")

compas_results_race = read.csv("/Users/jpgard/Documents/github/fair-robust/compas-racebs64e1000l1.0-race-results.csv")
compas_results_race_dp_S1z0.8 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/compas-racebs64lr0.01e1000l1.0dpTrueclip1.0z0.8-race-results.csv")
compas_results_race_dp_S10z0.4 = read.csv("/Users/jpgard/Documents/github/fair-robust/results/compas-racebs64lr0.01e1000l1.0dpTrueclip10.0z0.4-race-results.csv")

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


format_dp_df <- function(df_dp, clipnorm = 1.0, z = 0.8){
  dp_str = paste0("DP-SGD; Clipping Norm ", clipnorm, ", z = ", z)
  df_dp$dp = dp_str
  return(df_dp)
}
combine_results <- function(df_nodp, df_dp1, df_dp2){
  df_nodp$dp = "No DP"
  df_nodp$eps = NaN
  df_nodp$delta = NaN
  df_out = rbind(df_nodp, df_dp1, df_dp2)
  df_out$dp = factor(df_out$dp)
  df_out$dp = relevel(df_out$dp, "No DP")  # Make 'No DP' the first factor level.
  return(df_out)
}

german_results_dp_S1z0.8 <- format_dp_df(german_results_dp_S1z0.8)
adult_results_sex_dp_S1z0.8 <- format_dp_df(adult_results_sex_dp_S1z0.8)
adult_results_race_dp_S1z0.8 <- format_dp_df(adult_results_race_dp_S1z0.8)
compas_results_race_dp_S1z0.8 <- format_dp_df(compas_results_race_dp_S1z0.8)

german_results_dp_S10z0.4 <- format_dp_df(german_results_dp_S10z0.4, clipnorm = 10, z = 0.4)
adult_results_sex_dp_S10z0.4 <- format_dp_df(adult_results_sex_dp_S10z0.4, clipnorm = 10, z = 0.4)
adult_results_race_dp_S10z0.4 <- format_dp_df(adult_results_race_dp_S10z0.4, clipnorm = 10, z = 0.4)
compas_results_race_dp_S10z0.4 <- format_dp_df(compas_results_race_dp_S10z0.4, clipnorm = 10, z = 0.4)

german_results = combine_results(german_results, german_results_dp_S1z0.8, german_results_dp_S10z0.4)
adult_results_sex = combine_results(adult_results_sex, adult_results_sex_dp_S1z0.8, adult_results_sex_dp_S10z0.4)
adult_results_race = combine_results(adult_results_race, adult_results_race_dp_S1z0.8, adult_results_race_dp_S10z0.4)
compas_results_race = combine_results(compas_results_race, compas_results_race_dp_S1z0.8, compas_results_race_dp_S10z0.4)

# Plot for german results
german_results$train_subset <- forcats::fct_recode(german_results$train_subset, "Union"="all", 
                    "Minority"="minority", "Majority"="majority")
german_results$eval_subset <- forcats::fct_recode(german_results$eval_subset, "Union"="all", 
                                                   "Minority"="minority", "Majority"="majority")
p_german <- german_results %>% 
  data_summary(varname="loss", 
               groupnames=c("train_subset", "eval_subset", "dp")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
                ) +
  ggtitle("German Credit Dataset") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2.5)),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        strip.text = element_text(size=rel(1.25))) + # size of the facet label text)
  facet_grid(. ~ dp) +
  labs(caption ="Sensitive Attribute: Sex")
p_german

# Plot for adult results (sex)
adult_results_sex$train_subset <- forcats::fct_recode(adult_results_sex$train_subset, "Union"="all", 
                                                   "Minority"="minority", "Majority"="majority")
adult_results_sex$eval_subset <- forcats::fct_recode(adult_results_sex$eval_subset, "Union"="all", 
                                                  "Minority"="minority", "Majority"="majority")
p_adult_sex <- adult_results_sex %>%
  data_summary(varname="loss",
               groupnames=c("train_subset", "eval_subset", "dp")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("Adult Dataset") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2.5)),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        strip.text = element_text(size=rel(1.25))) + # size of the facet label text)
  facet_grid(. ~ dp) +
  labs(caption ="Sensitive Attribute: Sex")
p_adult_sex

# Plot for adult results (race)
adult_results_race$train_subset <- forcats::fct_recode(adult_results_race$train_subset, "Union"="all", 
                                                      "Minority"="minority", "Majority"="majority")
adult_results_race$eval_subset <- forcats::fct_recode(adult_results_race$eval_subset, "Union"="all", 
                                                     "Minority"="minority", "Majority"="majority")
p_adult_race <- adult_results_race %>% data_summary(varname="loss", 
                              groupnames=c("train_subset", "eval_subset", "dp")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("Adult Dataset") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2.5)),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        strip.text = element_text(size=rel(1.25))) + # size of the facet label text)) +
  facet_grid(. ~ dp) +
  labs(caption ="Sensitive Attribute: Race")
p_adult_race

# Plot for compas results (race)
compas_results_race$train_subset <- forcats::fct_recode(compas_results_race$train_subset, "Union"="all", 
                                                       "Minority"="minority", "Majority"="majority")
compas_results_race$eval_subset <- forcats::fct_recode(compas_results_race$eval_subset, "Union"="all", 
                                                      "Minority"="minority", "Majority"="majority")
p_compas_race <- compas_results_race %>% data_summary(varname="loss", 
                                    groupnames=c("train_subset", "eval_subset", "dp")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, fill=eval_subset)) +
  geom_col(position = position_dodge2(width = 0.4, preserve = "single"), colour = "black") +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("COMPAS Dataset") +
  xlab("Training Subset") +
  ylab("Loss") + 
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"), name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2.5)),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        strip.text = element_text(size=rel(1.25))) +
  facet_grid(. ~ dp) +
  labs(caption ="Sensitive Attribute: Race\nNote: 'Minority' attribute group is larger than 'Majority' in COMPAS.")
p_compas_race

grid.arrange(p_german, p_compas_race, p_adult_sex, p_adult_race, nrow=2, ncol=2)
library(ggpubr)
ggarrange(p_german, p_compas_race, p_adult_sex, p_adult_race, nrow=2, ncol=2, common.legend = TRUE, legend="bottom")
ggsave("./subset_experiments_fairness_datasets_dp.pdf", device="pdf", width=21, height=9)
