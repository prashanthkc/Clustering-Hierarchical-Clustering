########################Problem 1#############################################
install.packages("readxl")
library(readxl)
my_data <- read_excel("C:\\Users\\hp\\Desktop\\EastWestAirlines.xlsx" , sheet =2)
new_data <- my_data[ , c(2:12)]

summary(new_data)
sum(is.na(my_data))
sum(is.null(my_data))

dupl<- duplicated(new_data)
sum(dupl)
new_data <- new_data[!dupl , ]

norm_data <- scale(new_data[ , c(1:10)])
norm_data <- data.frame(norm_data)

#univariate analysis
#boxplot for all features of df new_data
#par(mfrow = c(2,2))
boxplot(new_data$Balance)
boxplot(new_data$Qual_miles)
boxplot(new_data$cc1_miles)
boxplot(new_data$cc2_miles)
boxplot(new_data$cc3_miles)
boxplot(new_data$Bonus_miles)
boxplot(new_data$Bonus_trans)
boxplot(new_data$Flight_miles_12mo)
boxplot(new_data$Flight_trans_12)
boxplot(new_data$Days_since_enroll)


qunt_balance <- quantile(new_data$Balance , probs = c(.25 , .75))
winso_balance <- quantile(new_data$Balance , probs = c(.01 , .93) , na.rm = TRUE)
H_balance <- 1.5*IQR(new_data$Balance , na.rm = TRUE)
new_data$Balance[new_data$Balance<(qunt_balance[1]-H_balance)] <- winso_balance[1]
new_data$Balance[new_data$Balance>(qunt_balance[2]+H_balance)] <- winso_balance[2]
boxplot(new_data$Balance)

qunt_Bonus_miles <- quantile(new_data$Bonus_miles , probs = c(.25 , .75))
winso_Bonus_miles <- quantile(new_data$Bonus_miles , probs = c(.05 , .93) , na.rm = TRUE)
H_Bonus_miles <- 1.5*IQR(new_data$Bonus_miles , na.rm = TRUE)
new_data$Bonus_miles[new_data$Bonus_miles<(qunt_Bonus_miles[1]-H_Bonus_miles)] <- winso_Bonus_miles[1]
new_data$Bonus_miles[new_data$Bonus_miles>(qunt_Bonus_miles[2]+H_Bonus_miles)] <- winso_Bonus_miles[2]
boxplot(new_data$Bonus_miles)


qunt_Bonus_trans <- quantile(new_data$Bonus_trans , probs = c(.25 , .75))
winso_Bonus_trans <- quantile(new_data$Bonus_trans , probs = c(.01 , .95) , na.rm = TRUE)
H_Bonus_trans <- 1.5*IQR(new_data$Bonus_trans , na.rm = TRUE)
new_data$Bonus_trans[new_data$Bonus_trans<(qunt_Bonus_trans[1]-H_Bonus_trans)] <- winso_Bonus_trans[1]
new_data$Bonus_trans[new_data$Bonus_trans>(qunt_Bonus_trans[2]+H_Bonus_trans)] <- winso_Bonus_trans[2]
boxplot(new_data$Bonus_trans)

qunt_Flight_miles_12mo <- quantile(new_data$Flight_miles_12mo , probs = c(.25 , .75))
winso_Flight_miles_12mo <- quantile(new_data$Flight_miles_12mo , probs = c(.01 , .85) , na.rm = TRUE)
H_Flight_miles_12mo <- 1.5*IQR(new_data$Flight_miles_12mo , na.rm = TRUE)
new_data$Flight_miles_12mo[new_data$Flight_miles_12mo<(qunt_Flight_miles_12mo[1]-H_Flight_miles_12mo)] <- winso_Flight_miles_12mo[1]
new_data$Flight_miles_12mo[new_data$Flight_miles_12mo>(qunt_Flight_miles_12mo[2]+H_Flight_miles_12mo)] <- winso_Flight_miles_12mo[2]
boxplot(new_data$Flight_miles_12mo)

qunt_Flight_trans_12 <- quantile(new_data$Flight_trans_12 , probs = c(.25 , .75))
winso_Flight_trans_12 <- quantile(new_data$Flight_trans_12 , probs = c(.01 , .85) , na.rm = TRUE)
H_Flight_trans_12 <- 1.5*IQR(new_data$Flight_trans_12 , na.rm = TRUE)
new_data$Flight_trans_12[new_data$Flight_trans_12<(qunt_Flight_trans_12[1]-H_Flight_trans_12)] <- winso_Flight_trans_12[1]
new_data$Flight_trans_12[new_data$Flight_trans_12>(qunt_Flight_trans_12[2]+H_Flight_trans_12)] <- winso_Flight_trans_12[2]
boxplot(new_data$Flight_trans_12)


#histogram for all features of df new_data 
hist(new_data$Balance)
hist(new_data$Qual_miles)
hist(new_data$cc1_miles)
hist(new_data$cc2_miles)
hist(new_data$cc3_miles)
hist(new_data$Bonus_miles)
hist(new_data$Bonus_trans)
hist(new_data$flight_miles_12mo)
hist(new_data$Flight_trans_12)
hist(new_data$Days_since_enroll)

#dotchart for all features of df new_data 
dotchart(new_data$Balance)
dotchart(new_data$Qual_miles)
dotchart(new_data$cc1_miles)
dotchart(new_data$cc2_miles)
dotchart(new_data$cc3_miles)
dotchart(new_data$Bonus_miles)
dotchart(new_data$Bonus_trans)
dotchart(new_data$flight_miles_12mo)
dotchart(new_data$Flight_trans_12)
dotchart(new_data$Days_since_enroll)

#bivariate analysis
#scatterplot for some features of df norm_data
install.packages("ggplot2")
library(ggplot2)
qplot(Balance,Qual_miles,data = new_data,color = cc1_miles,geom = "point")
qplot(Bonus_miles,Flight_trans_12,data = new_data,color = Flight_trans_12,geom = "point")
qplot(cc3_miles,cc1_miles,data = new_data,color = cc2_miles,geom = "point")

#Model Building

Edist <- dist(norm_data, method = "euclidean")

Hcl1 <- hclust(Edist , method = "complete")
Hcl2 <- hclust(Edist , method = "single")
Hcl3 <- hclust(Edist , method = "average")
Hcl4 <- hclust(Edist , method = "centroid")
#dendogram
plot(Hcl1 , hang= 1)
plot(Hcl2 , hang= -1)
plot(Hcl3 , hang= -1)
plot(Hcl4 , hang= -1)

clustered <- cutree(Hcl1,k=3)
rect.hclust(Hcl1, k = 3, border = "red")

Group <- as.matrix(clustered) 
final <- data.frame(Group , new_data)

aggregate(new_data[, 1:10], by = list(final$Group), FUN = mean) 

install.packages("readr")
library(readr)
write_csv(final, "final.csv")
getwd()

##############################Problem 2###################################
install.packages("readr")
library(readr)
crime_data <- read_csv("C:\\Users\\hp\\Desktop\\crime_data.csv")
new_crime_data <- crime_data[ , c(2:5)]

sum(is.na(new_crime_data))

summary(new_crime_data)

dup<- duplicated(new_crime_data)
sum(dup)
new_crime_data <- new_crime_data[!dup , ]
str(new_crime_data)

boxplot(new_crime_data$Murder)
boxplot(new_crime_data$Assault)
boxplot(new_crime_data$UrbanPop)
boxplot(new_crime_data$Rape)

qunt_Rape <- quantile(new_crime_data$Rape , probs = c(.25 , .75))
winso_Rape <- quantile(new_crime_data$Rape , probs = c(.01 , .95) , na.rm = TRUE)
H_Rape <- 1.5*IQR(new_crime_data$Rape , na.rm = TRUE)
new_crime_data$Rape[new_crime_data$Rape<(qunt_Rape[1]-H_Rape)] <- winso_Rape[1]
new_crime_data$Rape[new_crime_data$Rape>(qunt_Rape[2]+H_Rape)] <- winso_Rape[2]
boxplot(new_crime_data$Rape)


norm_crime_data <- scale(new_crime_data)
norm_crime_data <- as.data.frame(norm_crime_data)
summary(norm_crime_data)

#bivariate analysis
#scatterplot for some features of df norm_data
install.packages("ggplot2")
library(ggplot2)
qplot(Murder,Assault ,data = norm_crime_data,geom = "point")
qplot(UrbanPop,Rape,data = norm_crime_data,geom = "point")
qplot(Murder,UrbanPop,data = norm_crime_data,geom = "point")

dist_crime_data <- dist(norm_crime_data , method = "euclidean")

fit_crime_data <- hclust(dist_crime_data , method = "complete")
plot(fit_crime_data , hang = -1)

clust_crime_data <- cutree(fit_crime_data , k=3)
rect.hclust(fit_crime_data , k=3 , border = "green")

top_three <- as.matrix(clust_crime_data)
final_crime_data <- data.frame(top_three , crime_data)

aggregate(crime_data[, 2:5], by = list(final_crime_data$top_three), FUN = mean)

write_csv(final_crime_data , "final_crime_data.csv")
getwd()

#########################################Problem 3###########################
install.packages("readxl")
library(readxl)

telco_data <- read_excel("C:\\Users\\hp\\Desktop\\Telco_customer_churn.xlsx")
my_telco_data <-  telco_data[ , c(-1,-2,-3)]
sum(is.na(my_telco_data))

summary(my_telco_data)

dups <- duplicated(my_telco_data)
sum(dups)
my_telco_data <- my_telco_data[!dups , ]
str(my_telco_data)

boxplot(my_telco_data["Tenure in Months"])
boxplot(my_telco_data["Avg Monthly Long Distance Charges"])
boxplot(my_telco_data["Avg Monthly GB Download"])
boxplot(my_telco_data["Monthly Charge"])
boxplot(my_telco_data["Total Charges"])
boxplot(my_telco_data["Total Refunds"])
boxplot(my_telco_data["Total Extra Data Charges"])
boxplot(my_telco_data["Total Long Distance Charges"])
boxplot(my_telco_data["Total Revenue"])

qunt_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.25 , .75))
winso_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.01 , .93) , na.rm = TRUE)
H_Avg_Monthly_GB_Download <- 1.5*IQR(my_telco_data$"Avg Monthly GB Download" , na.rm = TRUE)
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download"<(qunt_Avg_Monthly_GB_Download[1]-H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[1]
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download">(qunt_Avg_Monthly_GB_Download[2]+H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[2]
boxplot(my_telco_data$"Avg Monthly GB Download")

qunt_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.25 , .75))
winso_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.01 , .92) , na.rm = TRUE)
H_Total_Refunds <- 1.5*IQR(my_telco_data$"Total Refunds" , na.rm = TRUE)
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds"<(qunt_Total_Refunds[1]-H_Total_Refunds)] <- winso_Total_Refunds[1]
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds">(qunt_Total_Refunds[2]+H_Total_Refunds)] <- winso_Total_Refunds[2]
boxplot(my_telco_data$"Total Refunds")

qunt_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.25 , .75))
winso_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.01 , .85) , na.rm = TRUE)
H_Total_Extra_Data_Charges <- 1.5*IQR(my_telco_data$"Total Extra Data Charges" , na.rm = TRUE)
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges"<(qunt_Total_Extra_Data_Charges[1]-H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[1]
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges">(qunt_Total_Extra_Data_Charges[2]+H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[2]
boxplot(my_telco_data$"Total Extra Data Charges")

qunt_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.25 , .75))
winso_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.01 , .95) , na.rm = TRUE)
H_Total_Long_Distance_Charges <- 1.5*IQR(my_telco_data$"Total Long Distance Charges" , na.rm = TRUE)
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges"<(qunt_Total_Long_Distance_Charges[1]-H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[1]
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges">(qunt_Total_Long_Distance_Charges[2]+H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[2]
boxplot(my_telco_data$"Total Long Distance Charges")

qunt_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.25 , .75))
winso_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.01 , .99) , na.rm = TRUE)
H_Total_Revenue <- 1.5*IQR(my_telco_data$"Total Revenue" , na.rm = TRUE)
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue"<(qunt_Total_Revenue[1]-H_Total_Revenue)] <- winso_Total_Revenue[1]
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue">(qunt_Total_Revenue[2]+H_Total_Revenue)] <- winso_Total_Revenue[2]
boxplot(my_telco_data$"Total Revenue")

install.packages("ggplot2")
library(ggplot2)
qplot(my_telco_data$"Total Revenue",my_telco_data$"Total Long Distance Charges" ,data = my_telco_data,geom = "point")
qplot(my_telco_data$"Total Charges",my_telco_data$"Total Refunds",data = my_telco_data,geom = "point")


install.packages("fastDummies")
library(fastDummies)

my_telco_data_dummy <- dummy_cols(my_telco_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

norm_telco_data <- scale(my_telco_data_dummy)
norm_telco_data <- as.data.frame(norm_telco_data)
summary(norm_telco_data)

dist_telco_data <- dist(norm_telco_data , method = "euclidean")
fit_telco_data <- hclust(dist_telco_data , method = "complete")
plot(fit_telco_data , hang = -1)

clust_telco_data <- cutree(fit_telco_data , k=3)
rect.hclust(fit_telco_data , k=3 , border = "green")

top_three_telco <- as.matrix(clust_telco_data)
final_telco_data <- data.frame(top_three_telco , telco_data)

aggregate(telco_data[,1:30], by = list(final_telco_data$top_three_telco), FUN = mean)
install.packages("readr")
library(readr)
write_csv(final_telco_data , "final_crime_data.csv")
getwd()

#daisy()
library(cluster)
telco_dist <- daisy(norm_telco_data , metric = "gower" )
summary(telco_dist)
telco_dist1 <- as.matrix(telco_dist)

fit_telco_data <- hclust(telco_dist , method = "complete")
plot(fit_telco_data , hang = -1)
clust_telco_data <- cutree(fit_telco_data , k = 3)
rect.hclust(fit_telco_data ,k=3 , border = "red")

###############################program 4###############################################
install.packages("readr")
library(readr)

auto_data <- read_csv("C:\\Users\\hp\\Desktop\\AutoInsurance.csv")
new_auto_data <- auto_data[-1]

sum(is.na(new_auto_data))

summary(new_auto_data)

dup<- duplicated(new_auto_data)
sum(dup)
new_auto_data <- new_auto_data[!dup , ]
str(new_auto_data)

boxplot(new_auto_data$`Customer Lifetime Value`)
boxplot(new_auto_data$Income)
boxplot(new_auto_data$`Monthly Premium Auto`)
boxplot(new_auto_data$`Months Since Last Claim`)
boxplot(new_auto_data$`Months Since Policy Inception`)
boxplot(new_auto_data$`Total Claim Amount`)

qunt_Customer_Lifetime_Value <- quantile(new_auto_data$`Customer Lifetime Value` , probs = c(.25 , .75))
winso_Customer_Lifetime_Value <- quantile(new_auto_data$`Customer Lifetime Value` , probs = c(.01 , .90) , na.rm = TRUE)
H_Customer_Lifetime_Value <- 1.5*IQR(new_auto_data$`Customer Lifetime Value` , na.rm = TRUE)
new_auto_data$`Customer Lifetime Value`[new_auto_data$`Customer Lifetime Value`<(qunt_Customer_Lifetime_Value[1]-H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[1]
new_auto_data$`Customer Lifetime Value`[new_auto_data$`Customer Lifetime Value`>(qunt_Customer_Lifetime_Value[2]+H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[2]
boxplot(new_auto_data$`Customer Lifetime Value`)

qunt_Monthly_Premium_Auto <- quantile(new_auto_data$`Monthly Premium Auto` , probs = c(.25 , .75))
winso_Monthly_Premium_Auto <- quantile(new_auto_data$`Monthly Premium Auto` , probs = c(.01 , .95) , na.rm = TRUE)
H_Monthly_Premium_Auto <- 1.5*IQR(new_auto_data$`Monthly Premium Auto` , na.rm = TRUE)
new_auto_data$`Monthly Premium Auto`[new_auto_data$`Monthly Premium Auto`<(qunt_Monthly_Premium_Auto[1]-H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[1]
new_auto_data$`Monthly Premium Auto`[new_auto_data$`Monthly Premium Auto`>(qunt_Monthly_Premium_Auto[2]+H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[2]
boxplot(new_auto_data$`Monthly Premium Auto`)

qunt_Total_Claim_Amount <- quantile(new_auto_data$`Total Claim Amount` , probs = c(.25 , .75))
winso_Total_Claim_Amount <- quantile(new_auto_data$`Total Claim Amount` , probs = c(.01 , .95) , na.rm = TRUE)
H_Total_Claim_Amount <- 1.5*IQR(new_auto_data$`Total Claim Amount` , na.rm = TRUE)
new_auto_data$`Total Claim Amount`[new_auto_data$`Total Claim Amount`<(qunt_Total_Claim_Amount[1]-H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[1]
new_auto_data$`Total Claim Amount`[new_auto_data$`Total Claim Amount`>(qunt_Total_Claim_Amount[2]+H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[2]
boxplot(new_auto_data$`Total Claim Amount`)


install.packages("ggplot2")
library(ggplot2)
qplot(new_auto_data$Income,new_auto_data$"Monthly Premium Auto" ,data = new_auto_data,geom = "point")
qplot(new_auto_data$"Total Claim Amount",new_auto_data$"Months Since Policy Inception",data = new_auto_data,geom = "point")


install.packages("fastDummies")
library(fastDummies)

my_auto_data_dummy <- dummy_cols(new_auto_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

norm_auto_data <- scale(my_auto_data_dummy)
summary(norm_auto_data)

dist_auto_data <- dist(norm_auto_data , method = "euclidean")

fit_auto_data <- hclust(dist_auto_data , method = "complete")
plot(fit_auto_data , hang = -1)

clust_auto_data <- cutree(fit_auto_data , k=3)
rect.hclust(fit_auto_data , k=3 , border = "green")

auto_top_three <- as.matrix(clust_auto_data)
final_auto_data <- data.frame(auto_top_three , new_auto_data)

aggregate(final_auto_data[, 1:24], by = list(final_auto_data$auto_top_three), FUN = mean)

write_csv(final_auto_data , "final_auto_data.csv")
getwd()
