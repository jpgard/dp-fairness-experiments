library(ggplot2)
library(ggthemes)
library(magrittr)
library(gridExtra)

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
p_german <- german_results %>% 
  data_summary(varname="loss", 
               groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, col=eval_subset)) +
  geom_bar(stat="identity", 
           fill="grey",
           position=position_dodge(.9),
           width=0.5,
           size=1) +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
                ) +
  ggtitle("German Credit Dataset\nSensitive Attribute: Sex\n1 s.d. error bars shown (n = 1000)") +
  xlab("Training Subset") +
  scale_color_colorblind(name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw()
p_german

# Plot for adult results (sex)
p_adult_sex <- adult_results_sex %>%
  data_summary(varname="loss",
               groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, col=eval_subset)) +
  geom_bar(stat="identity", 
           fill="grey",
           position=position_dodge(.9),
           width=0.5,
           size=1) +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("Adult Dataset\nSensitive Attribute: Sex\n1 s.d. error bars shown (n = 37,000)") +
  xlab("Training Subset") +
  scale_color_colorblind(name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw()
p_adult_sex

# Plot for adult results (race)
p_adult_race <- adult_results_race %>% data_summary(varname="loss", 
                              groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, col=eval_subset)) +
  geom_bar(stat="identity", 
           fill="grey",
           position=position_dodge(.9),
           width=0.5,
           size=1) +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("Adult Dataset\nSensitive Attribute: Race\n1 s.d. error bars shown (n = 37,000)") +
  xlab("Training Subset") +
  scale_color_colorblind(name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw()
p_adult_race

# Plot for compas results (race)
p_compas_race <- compas_results_race %>% data_summary(varname="loss", 
                                    groupnames=c("train_subset", "eval_subset")) %>%
  ggplot(aes(x=train_subset, y=loss, group=eval_subset, col=eval_subset)) +
  geom_bar(stat="identity", 
           fill="grey",
           position=position_dodge(.9),
           width=0.5,
           size=1) +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
  ) +
  ggtitle("COMPAS Dataset\nSensitive Attribute: Race\n1 s.d. error bars shown (n = 5,200)\nNote: 'Minority' group is LARGER in COMPAS") +
  xlab("Training Subset") +
  scale_color_colorblind(name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw()
p_compas_race

grid.arrange(p_german, p_compas_race, p_adult_sex, p_adult_race, ncol=2)
