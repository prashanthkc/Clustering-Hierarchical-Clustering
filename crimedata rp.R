
install.packages("readr")
install.packages("readxl")
library(readxl)
library(readr)

crime_data <- read.csv(file.choose())
is.na(crime_data)
sum(is.na(crime_data))

my_data <- crime_data[2:5]
summary(my_data)

#normalization
normalized_data <- scale(my_data)
summary(normalized_data)

#distance matrix
d <- dist(normalized_data , method = "euclidean")

hep <- hclust(d,method = "complete")

#disply dendogram
plot(hep)
plot(hep , hang = -1)

group <- cutree(hep , k=3)

rect.hclust(hep , k=3 , border = "green")

membership <-as.matrix(group)
final <- data.frame(membership , my_data)

aggregate(my_data[ ,2:5 ] , by = list(final$membership), FUN = mean)
write.csv(final,"crime_data.csv")
getwd()
 