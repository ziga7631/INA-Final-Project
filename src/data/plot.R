install.packages('ggplot2')
install.packages('reshape2')

# Load the library
library(ggplot2)
install.packages("RColorBrewer")
library(RColorBrewer)

install.packages("patchwork")
library(patchwork)


# Create a data frame
df <- data.frame(
  model = c("CF", "Neural net", "PPR bi", "PPR multi", "TSPR bi", "TSPR multi"),
  value = c(c(0.00000923,0.00000543,0.00018337,0.00008091,0.00017493,0.00017493),  # generates 7 random precision values between 0.7 and 1
            c(0.00125178,0.00017611,0.00739943,0.00831558,0.00698128,0.00698128)),  # generates 7 random recall values between 0.6 and 1
  metric = rep(c("Precision", "Recall"), each = 6)
)
# Melt the data frame to long format
df_long <- reshape2::melt(df, id.vars = "model")

# Define color palette
my_colors <- c("#ff5a5f", "#3993dd")

# Plot 1: Precision
p1 <- ggplot(df[df$metric == "Precision",], aes(x=model, y=value, fill=metric)) +
  geom_bar(stat="identity", position=position_dodge(), color="black", size=0.5) +
  scale_fill_manual(values=my_colors[1], guide = FALSE) +
  theme_minimal() +
  labs(x = "Model", y = "Value", fill = "Metric", 
       title = "Precision@k") +
  theme(plot.title = element_text(hjust = 0.5), 
        panel.border = element_blank(), 
        axis.line = element_line(color = "black"),
        legend.position = "none")

# Plot 2: Recall
p2 <- ggplot(df[df$metric == "Recall",], aes(x=model, y=value, fill=metric)) +
  geom_bar(stat="identity", position=position_dodge(), color="black", size=0.5) +
  scale_fill_manual(values=my_colors[2], guide = FALSE) +
  theme_minimal() +
  labs(x = "Model", y = "Value", fill = "Metric", 
       title = "Recall@k") +
  theme(plot.title = element_text(hjust = 0.5), 
        panel.border = element_blank(), 
        axis.line = element_line(color = "black"),
        legend.position = "none")

p1 / p2
  
  
  
  
  
  
  
  
  
  
  
  
  
  install.packages('ggplot2')
  install.packages('reshape2')
  
  # Load the library
  library(ggplot2)
  install.packages("RColorBrewer")
  library(RColorBrewer)
  
  # Create a data frame
  df <- data.frame(
    model = c("CF", "Neural net", "PPR bi", "PPR multi", "TSPR bi", "TSPR multi"),
    precision = c(0.00000923,0.00000543,0.00018337,0.00008091,0.00017493,0.00017493),  # set precision values to 1
    recall =  c(0.00125178,0.00017611,0.00739943,0.00831558,0.00698128,0.00698128)  # set recall values to 1
  )
  # Melt the data frame to long format
  df_long <- reshape2::melt(df, id.vars = "model")
  
  # Define color palette
  my_colors <- c("#ff5a5f", "#3993dd")
  
  # Create a bar plot
  ggplot(df_long, aes(x=model, y=value, fill=variable)) +
    geom_bar(stat="identity", position=position_dodge(), color="black", size=0.5) +
    scale_fill_manual(values=my_colors, labels=c("precision@k", "recall@k")) +
    theme_minimal() +
    labs(x = "Model", y = "Value", fill = "Metric", 
         title = "Model Precision@k and Recall@k Results") +
    theme(plot.title = element_text(hjust = 0.5), 
          panel.border = element_blank(), 
          axis.line = element_line(color = "black"), 
          legend.position = "bottom", 
          legend.direction = "horizontal", 
          legend.title = element_blank(), 
          legend.box.spacing = unit(0.2, "cm"), 
          legend.spacing = unit(0.1, "cm") +
            scale_y_log10()  # Add this line for a log scale on the y-axis