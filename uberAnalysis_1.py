import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf

#Reading the csv file from my local disc D
uber_17 = pd.read_csv(r'D:\python\pycharm\uberAnalysis\Data\uber_2017.csv')
uber_17

import pandas as pd
uber_18 = pd.read_csv(r"D:\python\pycharm\uberAnalysis\Data\UBER_2018_1.csv")
uber_18

#using to_strings to read the whole dataframe in the console
print(uber_18.to_string())

#Mearging my two dataframes into one csv file -uber_1718
uber_1718 = uber_17.append(uber_18)

uber_1718.to_csv(r"D:\python\pycharm\uber_1738.csv" , index = True)

#Changing paths in the files


import csv

test = pd.read_csv(r"D:\python\pycharm\uber_1738.csv" )

test.to_csv(r"D:\python\pycharm\uber_1739.csv" , index = True)

#cleaning empty cells
uber_1718

#Getting the dataframe overall details
uber_1718.describe()


#getting a list of all column names
my_list = uber_1718.columns.values.tolist()
print(my_list)

#Drop trip ID,request date,request time UTC,Request Timezone offset from UTC
#request time UTC,Drop off Time (UTC),Email,Unnamed: 27,28,29

uber_1718_new2 = uber_1718.drop(['Trip ID', 'Request Date', 'Request Time (UTC)' ,
                                'Request Timezone offset from UTC', 'Drop off Time (UTC)', 'Email','Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28']
                                , axis=1)

my_list3 = uber_1718_new2.columns.values.tolist()
print(my_list3)

uber_1718_new2.describe()

uber_1718_new2.isnull()

pd.set_option("display.max_columns" , None)
uber_1718_new2.isnull()

#checking for empty cells
uber_1718_new2.isnull().sum()
uber_1718_new2.dtypes

#Fixing the empty cells
#drop off date
'''uber_1718_new2['Drop Off Date'].fillna(, inplace = True)

for x in uber_1718_new2:
    if x['Drop Off Date'].isnull() = True:
        x['Drop Off Date'] = x[request Date (Local)]'''


uber_1718_new2['Drop Off Date'] = uber_1718_new2['Drop Off Date'].fillna(uber_1718_new2['Request Date (Local)'])
uber_1718_new2.isnull().sum()

#Fixing first name and last name empty cells
#dropping the nulls in first name column
uber_1718_new3 = uber_1718_new2.dropna(axis = 0, subset = ['First Name'])
uber_1718_new3.isnull().sum()

#dropping the null in both the first name and the second name column
uber_1718_new4 = uber_1718_new2.dropna( axis = 0 ,subset = ['First Name' , 'Last Name' ])
uber_1718_new4.isnull().sum()


uber_1718_new4.to_csv(r'D:\python\pycharm\uber_1718_new4.csv' , index = True)

uber_1718_news5 = pd.read_csv(r'D:\python\pycharm\uber_1718_new4.csv')
uber_1718_news5.isnull().sum()
uber_1718_news5['Expense Code'].unique()

#converting all the values in the column section to date format
uber_1718_news5['Request Date (Local)'] = pd.to_datetime(uber_1718_news5['Request Date (Local)'])
print(uber_1718_news5.to_string())

uber_1718_news5['Drop Off Date'] = pd.to_datetime(uber_1718_news5['Drop Off Date'])

#Grouping the data by month titles.. EDA
uber_1718_groupedM = uber_1718_news5.groupby(pd.Grouper(key = 'Request Date (Local)' , axis = 0,freq = 'M')).sum().reset_index()
print(uber_1718_groupedM.to_string())

#Grouping the data by month titles.. changed date format for analysis
#uber_1718_groupedMformated = pd.to_datetime(uber_1718_groupedM['Request Date (Local)']).dt.strftime('%Y-%b')
uber_1718_groupedMcopy = uber_1718_groupedM.copy()
uber_1718_groupedMcopy['Request Date (Local)'] = pd.to_datetime(uber_1718_groupedMcopy['Request Date (Local)']).dt.strftime('%Y-%b')

#from the grouped data, visualize monthly total charges (bar graphs), Monthly time spent in commute(bar graphs),
#Monthly distance covered vs time taken.(line graph)

#monthly total charges (bar graphs)
#Total charges per month are generally increasing over the two years with DEC 2018 having the highest charges
#There is no indication of seasonality
#EDA
x = uber_1718_groupedM['Request Date (Local)']
y = uber_1718_groupedM['Total Charge in KES']

plt.bar(x,y,width = 20)
plt.xticks(x, rotation = 90)
plt.xlabel('Months')
plt.ylabel('Total Charges(KES)')
plt.title('Total charges per month')
plt.show()

#total charges per month with date formated in year-date format
#Creating a new dataframe for 2017 and 2018 separately
year2017 = uber_1718_groupedMcopy[pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year == 2017]
year2018 = uber_1718_groupedMcopy[pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year == 2018]

#total charges per month with date formated in year-date format combine 2017 and 2018
x1 = uber_1718_groupedMcopy['Request Date (Local)']
y2 = uber_1718_groupedMcopy['Total Charge in KES']
#year2017s = (pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year == 2017).any()
#year2018s = (pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year == 2018).any()
#colors = ['Green' if (pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year == 2017).any() else (pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year != 2017).any()]
#colors3 = ['Green' if uber_1718_groupedMcopy['Request Date (Local)'] == year2017s else 'red' for s in uber_1718_groupedMcopy['Request Date (Local)']]
plt.bar(x1,y2,)
plt.xticks(x1, rotation = 90)
plt.xlabel('Months')
plt.ylabel('Total Charges(KES)')
plt.title('Total charges per month')
plt.show()

#visualization distinguishing between 2017 trends and 2018 trends  in a bar graph
year2017 = uber_1718_groupedMcopy[pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year == 2017]
year2018 = uber_1718_groupedMcopy[pd.to_datetime((uber_1718_groupedMcopy['Request Date (Local)'])).dt.year == 2018]
a19 = year2017['Request Date (Local)']
b19 = year2017['Total Charge in KES']
plt.bar(a19,b19, color = 'yellow')
plt.xticks( rotation = 90)

c19 = year2018['Request Date (Local)']
d19 = year2018['Total Charge in KES']
plt.bar(c19,d19, color = 'Green')
plt.xticks(rotation = 90)

plt.xlabel('Total Charge in KES')
plt.ylabel('Months')
plt.title('Total charges per Month')
plt.show()

#side by side comparison of months total charges
x21 =['January', 'February', 'March', 'April', 'May', 'June', 'july', 'August', 'September','October','November','December']
y201 = year2017['Total Charge in KES']
y202 = year2018['Total Charge in KES']
X_axis2 = np.arange(len(x21))
plt.bar(X_axis2 - 0.2, y201,0.4,label = '2017')
plt.bar(X_axis2 - 0.4, y202,0.4,label = '2018')
plt.xticks(X_axis2, x21)
plt.xlabel("Mothly period")
plt.ylabel("Distance(miles)")
plt.title("Distance travelled per month(2017 & 2018)")
plt.legend()
plt.show()

#Monthly time spent in commute(bar graphs). Time spent is generally increasing over the two years
#DEC 2018 has the highest duration spent in transit
#EDA
a = uber_1718_groupedM['Request Date (Local)']
b = uber_1718_groupedM['Duration (min)']

plt.bar(a,b , width = 20)
plt.xticks(x, rotation = 90)
plt.xlabel('Months')
plt.ylabel('Time(Minutes)')
plt.title('Time spent on commute')
plt.show()

#Frequency of duration travelled per month,periodic comparison between 2017 and 2018
x20 =['January', 'February', 'March', 'April', 'May', 'June', 'july', 'August', 'September','October','November','December']
y201 = year2017['Duration (min)']
y202 = year2018['Duration (min)']
X_axis = np.arange(len(x20))
plt.bar(X_axis - 0.2, y201,0.4,label = '2017')
plt.bar(X_axis - 0.4, y202,0.4,label = '2018')
plt.xticks(X_axis, x20)
plt.xlabel("Mothly period")
plt.ylabel("Duration(min)")
plt.title("Duration travelled per month(2017 & 2018)")
plt.legend()
plt.show()

#Distance trends over the two years.(line graph). The purpose is to observe the relationship
#between time periods and total milage covered.The graph shows an increasing need uber for services as time elapses
#however, there is no indication of seasonality and the relationship could be as a result of increasing client base
c = uber_1718_groupedM['Request Date (Local)']
d = uber_1718_groupedM['Distance (mi)']

plt.scatter(c,d)
plt.xticks(x, rotation=90)
plt.xlabel('Months')
plt.ylabel('Distance (mi)')
plt.title('Milage covered per month')
plt.show()

#a line graph of miles coveres vs Monthly period to provide a visual presentation of didtance trends
plt.plot(c,d,linestyle = 'dotted')
plt.xlabel('Months')
plt.ylabel('Distance (mi)')
plt.title('Miles Covered per Month')
plt.show()

#Histogram - x axis (hours) y - number of trips...line 243
#created an extra column with 1 to represent one trip per row
uber_1718_news5['No of trips'] = '1'
print(uber_1718_news5.to_string())
my_list1 = uber_1718_news5.columns.values.tolist()
print(my_list1)
uber_1718_news5.to_csv(r"D:\python\pycharm\uber_1718_newsT2.csv", index=True)



"""uber_1718_groupedMT = uber_1718_news5.groupby(pd.Grouper(key = 'Request Date (Local)' , axis = 0,)).sum().reset_index()
print(uber_1718_groupedMT.to_string())
uber_1718_groupedMT.to_csv(r"D:\python\pycharm\uber_1718_groupedMTp.csv", index=True)"""

#Separate hours from minutes . done on line 243

#number of trips per month
#Dec 2018 has the hisghest number of trips, The number of trips per month are generally increasing
uber_1718_news5['No of trips'] = uber_1718_news5['No of trips'].astype(int)
uber_1718_groupedM2 = uber_1718_news5.groupby(pd.Grouper(key = 'Request Date (Local)' , axis = 0,freq = 'M')).sum().reset_index()
print(uber_1718_groupedM2.to_string())

g = uber_1718_groupedM2['Request Date (Local)']
h = uber_1718_groupedM2['No of trips']

plt.bar(g, h, width=20)
plt.xticks(x, rotation=90)
plt.xlabel('Months')
plt.ylabel('No of trips')
plt.title('Number of trips per month')
plt.show()



#reasons for travel
uber_1718_news5["Expense Code"].unique()
#change client support,support,Support = SUPPORT
#National bank kenyatta Avenue,nssf settlement,nssf,1517,nakumatt lifestyle,follow up on nssf,payment of PAYE and NHIF,nssf follow up
#payment of PAYE and NHIF,from bank to office = OFFICE ADMIN
#work late,working late,WORK LATE = WORKLATE
#Meeting,meeting = MEETING
# a total of 5 expense codes, visualize expenditure per code expense
#identify the modal expense code
uber_1718_news6 = uber_1718_news5
uber_1718_news7 = uber_1718_news5

#Modify the expense codes to unique values
'''for x in range(len(uber_1718_news7["Expense Code"])):
    if uber_1718_news7["Expense Code"][x] in ['client support', 'support', 'Support']:
        uber_1718_news7["Expense Code"][x] = 'SUPPORT'
    else if uber_1718_news7["Expense Code"][x] in ['National bank kenyatta Avenue','nssf settlement','nssf','1517','nakumatt lifestyle','follow up on nssf']:
        uber_1718_news7["Expense Code"][x] = 'OFFICE ADMIN'
    #else uber_1718_news7["Expense Code"][x]  in ['payment of PAYE and NHIF','nssf follow up','payment of PAYE and NHIF','from bank to office']:
        #uber_1718_news7["Expense Code"][x] = 'OFFICE ADMIN'
    else uber_1718_news7["Expense Code"][x] in ['work late','working late','WORK LATE']:
        uber_1718_news7["Expense Code"][x] = 'WORKLATE'
    #elif uber_1718_news7["Expense Code"][x]  in ['Meeting','meeting']:
        #uber_1718_news7["Expense Code"][x] = 'MEETING'
    else:
        uber_1718_news7["Expense Code"][x]
uber_1718_news6["Expense Code"].unique()'''


uber_1718_news10 = pd.read_csv(r"D:\python\pycharm\uber_1718_news81.csv")

df = uber_1718_news10.replace(dict.fromkeys(['client support','support','Support'] , 'SUPPORT'))
df["Expense Code"].unique()

df2 = df.replace(dict.fromkeys(['National bank kenyatta Avenue','nssf settlement','nssf','1517','nakumatt lifestyle','follow up on nssf'] ,'OFFICE ADMIN'))

df3 = df2.replace(dict.fromkeys(['payment of PAYE and NHIF','nssf follow up', 'payment of PAYE and NHIF','from bank to office'], 'OFFICE ADMIN'))

df4 = df3.replace(dict.fromkeys(['work late','working late','WORK LATE'], 'WORKLATE'))

uber_1718_news11 = df4.replace(dict.fromkeys(['Meeting','meeting'], 'MEETING'))

uber_1718_news11["Expense Code"].unique()

#uber_1718_news8.to_csv(r"D:\python\pycharm\uber_1718_news81.csv", index=True)
uber_1718_news11['Request Date (Local)'] = pd.to_datetime(uber_1718_news11['Request Date (Local)'])



G2 = uber_1718_news11.groupby(["Expense Code" ]).agg({'No of trips' : ['sum'] ,'Total Charge in KES':['sum']}   ).reset_index()
G2.columns

uber_1718_news11_Bytecode = uber_1718_news11.groupby(["Expense Code"])['Total Charge in KES', 'No of trips'].sum().reset_index()

#A pie plot to show the proportions of number of trips per expense code
#Travel for client support takes the largest chunk of number of trips
i = uber_1718_news11_Bytecode['No of trips']
mylables = uber_1718_news11_Bytecode['Expense Code']
plt.pie(i,labels = mylables,autopct='%1.0f%%')
plt.title('Proportion of Number of Trips per expense code')
plt.show()

#A bar chart to illustrate total charges per expense code
#Support accounts for the highest expenditure followed by worklate
j = uber_1718_news11_Bytecode['Expense Code']
k = uber_1718_news11_Bytecode['Total Charge in KES']

plt.bar(j, k)
plt.xlabel('Expense Code')
plt.ylabel('Total Charge in KES')
plt.title('Total charges per expense code')
plt.show()

#Distance statistics
#Frequency of requests by time
import datetime
uber_1718_news11['Hours'] = (pd.to_datetime(uber_1718_news11['Request Time (Local)']).dt.hour)
#uber_1718_news11['Request Time (Local)'] = pd.to_datetime(uber_1718_news11['Request Time (Local)']).dt.time

#A histogram to visualize frequency of requests by time. Most rides are requested at 2100hours. Most rides
#are concentrated in the time range of 0900 hours and 1400hours
l = uber_1718_news11['Hours']
plt.hist(l , bins = [0,1,2,3,4,5,6,7,8,9 ,10 ,11 ,12, 13 ,14 ,15, 16 ,17 ,18 ,19 ,20 ,21 ,22, 23, 24] ,color='purple' ,edgecolor='black')
plt.xlabel('Hours of day')
plt.ylabel('No. of trips')
plt.title('Trip distribution per hour')
plt.show()

#Department statistics
#Assign a department to each employee, separate file provided.group and sum by department
# Frequency of travel per department, create a column (number of trips) with integer 1 and use group by dept then sum
#the number of trips column

my_list8 = uber_1718_news11.columns.values.tolist()
print(my_list8)
uber_1718_news11['FirstandSecondname'] = uber_1718_news11['First Name'] + ' ' + uber_1718_news11['Last Name']

dept_allocation1 = pd.read_csv(r'D:\python\pycharm\EmployeeNames.csv')


#extract employee names from dept_allocation1
Names = list(dept_allocation1['Employee Name'])

#creating an empty list that we will add department names
Department = []

#for loop to match department index from both datasets
for i in range(len(uber_1718_news11['FirstandSecondname'])):
    if uber_1718_news11['FirstandSecondname'][i] in Names:
        idx = Names.index(uber_1718_news11['FirstandSecondname'][i])
        Department.append(dept_allocation1['Department'][(idx + 1)])


print((pd.DataFrame(Department)).head())
uber_1718_news11["Department"] = Department
#Number of trips per department
#Quality assurance has the highest number of trips
uber_1718_news11_Deptgroup = uber_1718_news11.groupby(["Department"])['Total Charge in KES', 'No of trips'].sum().reset_index()

q = uber_1718_news11_Deptgroup["Department"]
r = uber_1718_news11_Deptgroup['No of trips']

plt.barh(q, r)
#plt.xticks(q, rotation=45)
plt.xlabel('No of trips')
plt.ylabel("Department")
plt.title('Number of Trips per Department')
plt.show()

#Total charges per department
#quality assurance has the highest total charges, consistent to having highest number of trips
s = uber_1718_news11_Deptgroup["Department"]
t = uber_1718_news11_Deptgroup['Total Charge in KES']

plt.bar(s, t)
plt.xticks(s, rotation=90)
plt.xlabel("Department")
plt.ylabel('Total Charge in KES')
plt.title('Total Charges per Department')
plt.show()

#employee statistics

grp_1 = uber_1718_news11.groupby(['FirstandSecondname']).agg({'No of trips': ['sum'], 'Total Charge in KES': ['sum']}).reset_index()
#grp_1sorted = grp_1.sort_values('Total Charge in KES' , axis=0, ascending=True)

#Correlation
#Total charges is strongly positively correlated to distance (o.86)and duration (0.71) of trip
Corr_1 = uber_1718_news11.corr(method='pearson')

print(Corr_1.to_string())

#visual representation of correlation in total charges,distance and duration.
#correlation between distance and total charges appears to be more linear
sns.pairplot(data=uber_1718_news11[['Distance (mi)', 'Duration (min)','Total Charge in KES']])

#A linearplot of distance and total charges
sns.lmplot(x='Distance (mi)', y='Total Charge in KES', data=uber_1718_news11)

#Training and Testing Data.Set a variable X equal to the numerical features of the uberdata and a variable Y equal to the total charge in KES column.
x = uber_1718_news11[['Distance (mi)','Duration (min)']]
x.head()
y = uber_1718_news11['Total Charge in KES']
y.head()

#split the data into training and testing sets. Set test_size=0.3 and random_state=101
#use model_selection.train_test_split from sklearn

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=101)

#Train the model from the training data
from sklearn.linear_model import LinearRegression

#create an instance of linear regression model
lm = LinearRegression()

#Train/fit on the training data
lm.fit(X_train,Y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# The coefficients/slopes of mode
#The coefficient is 45.10 for distance and 4.12 for duration
#All parameters kept constant 1 unit increase in distance will result to KES 45.01 in total charges
#All parameters kept constant 1 unit increase in duration will result to KES 4.12 in total charges
print(lm.coef_)


#Predicting Test Data
prediction = lm.predict(X_test)

# create a scatterplot of the real test values versus the predicted values to check the performance of our model
plt.scatter(Y_test,prediction, c='Green')
plt.title("Real test values versus the Predicted values")
plt.xlabel("Real test values")
plt.ylabel("Predicted values")
plt.show()

y = a +bx+cy

my_list5 = uber_1718_news11.columns.values.tolist()
print(my_list5)

uber1718regression = uber_1718_news11[['Total Charge in KES','Distance (mi)']]

#defining the variable
xr= uber1718regression['Distance (mi)'].tolist()
yr = uber1718regression['Total Charge in KES'].tolist()

#adding the constant term
xr = sm.add_constant(xr)

#performing the regression and #fitting the model
result = sm.OLS(yr,xr).fit()

#printing the summary table
print(result.summary())
































