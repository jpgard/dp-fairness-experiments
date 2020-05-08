library(ggplot2)
results = read.csv("/Users/jpgard/Documents/github/fair-robust/results.csv")

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

df2 <- data_summary(results, varname="loss", 
                    groupnames=c("train_subset", "eval_subset"))
# Convert dose to a factor variable
df2$dose=as.factor(df2$dose)
head(df2)

ggplot(data=df2, aes(x=train_subset, y=loss, group=eval_subset, col=eval_subset)) +
  geom_bar(stat="identity", fill="grey",
           position=position_dodge(.9),
           width=0.5) +
  geom_errorbar(aes(ymin=loss-sd, ymax=loss+sd), width=.2, size=1,
                position=position_dodge(.9)
                ) +
  ggtitle("German Credit Dataset\n1 s.d. error bars shown (n = 1000)") +
  xlab("Training Subset") +
  scale_color_discrete(name="Test Subset") +
  guides(color=guide_legend(override.aes=list(fill=NA))) +
  theme_bw()
