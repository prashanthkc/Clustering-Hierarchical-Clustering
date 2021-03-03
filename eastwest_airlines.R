
install.packages("readxl")
library(readxl)
library(readr)

my_data <- read.csv(file.choose(),2)
is.na(my_data)  #for na values
sum(is.na(my_data)) #sum of na values

new_data <- my_data[2:12]
summary(new_data)

#normalization 
normalized_data <- scale(new_data)
summary(normalized_data)

#distance matrix
 
d <- dist(normalized_data , method = "euclidean")

hep <- hclust(d,method = "complete")
hep1 <- hclust(d , method = "single")
hep2 <- hclust(d , method = "average")
hep3 <- hclust(d , method = "centroid")

#disply dendogram
plot(hep)
plot(hep , hang = -1)
plot(hep1)
plot(hep1 , hang = -1)

group <- cutree(hep , k=3)
rect.hclust(hep , k =3, border = "red")

membership <- as.matrix(group)
final <- data.frame(membership , new_data)

a = aggregate(new_data[ , 2:11], by=list(final$membership), FUN = mean)
write.csv(final,"eastwestairlines")
getwd()
