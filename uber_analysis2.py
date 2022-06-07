import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

uberAugApril = pd.read_csv(r'D:\python\pycharm\uberAnalysis\Data\uber_AugApril22.csv')
uberJanJuly = pd.read_csv(r'D:\python\pycharm\uberAnalysis\Data\uberJanJuly21.csv')

uberMerged = pd.concat([uberJanJuly,uberAugApril],ignore_index = True)
print(uberMerged.to_string())
uberMerged.columns

uberMerged.to_csv(r"D:\python\pycharm\uberAnalysis\Data\uberMerged.csv" , index = True)

uberMerged2 = uberMerged.drop(['Trip.Eats.ID',  'Request date (UTC)', 'Request Date (Local)',
       'Request Time (Local)', 'Drop-off date (local)','Drop-off time (local)',
       'Request time (UTC)','Drop-off date (UTC)', 'Drop-off time (UTC)','Request timezone offset from UTC','Invoices',
       'Programme','Group','Fare in local currency (excl. taxes)', 'Taxes in local currency',
       'Tip in local currency','Transaction amount in local currency (incl. taxes)','Local currency code',
        'Fare in KES (excl. taxes)','Taxes in KES','Tip in KES'], axis=1)

uberMerged2.columns

#Data exploration
print(uberMerged2.to_string())
uberMerged2.isnull().sum() #there are no nulls despite having values missing in some rows
uberMerged2.dtypes   #change date and time columns into date format
uberMerged2['Transaction type'].describe()
uberMerged2['Drop-off time (local)'].unique()
print(uberMerged2['Transaction amount in KES (incl. taxes)'].head(10))

#Converting transaction time stamp column to date-time format
uberMerged2['Transaction timestamp (UTC)'] = pd.to_datetime(uberMerged['Transaction timestamp (UTC)']).dt.strftime('%Y-%m-%d')
uberMerged2['Transaction timestamp (UTC)'] = pd.to_datetime(uberMerged['Transaction timestamp (UTC)']).dt.month_name()
uberMerged2['Transaction timestamp (UTC)'].dtypes

#Combining the first name and the surname to the full name column
uberJanJuly = uberMerged2['Fullname'] = uberMerged['First name'] + ' ' + uberMerged['Surname']

uberMergedNameModified = uberMerged2.drop(['First name', 'Surname'] , axis = 1)
uberMergedNameModified.columns

#Converting transaction time stamp column to date-time format
uberMergedNameDateModified = pd.read_csv(r"D:\python\pycharm\uberAnalysis\Data\uberMergedNameModified.csv")
uberMergedNameDateModified['Transaction timestamp (UTC)'] = pd.to_datetime(uberMergedNameDateModified['Transaction timestamp (UTC)']).dt.strftime('%Y-%m-%d')


#creating new columns for Month day and year to aid in analysis
uberMergedNameDateModified['Month'] = (pd.to_datetime(uberMergedNameDateModified['Transaction timestamp (UTC)']).dt.month)
uberMergedNameDateModified['Day'] = (pd.to_datetime(uberMergedNameDateModified['Transaction timestamp (UTC)']).dt.day_name())
uberMergedNameDateModified['Year'] = (pd.to_datetime(uberMergedNameDateModified['Transaction timestamp (UTC)']).dt.year)
uberMergedNameDateModified['Month Year'] = pd.to_datetime(uberMergedNameDateModified['Transaction timestamp (UTC)']).dt.strftime('%Y-%b')


#checking distribution using a scatter graph. Findings: There are a number of large transactions that are less than zero(-ves)
uberMergedNameDateModified[['Transaction timestamp (UTC)','Transaction amount in KES (incl. taxes)']].plot(kind = 'scatter', x='Transaction timestamp (UTC)',y='Transaction amount in KES (incl. taxes)')
uberMergedNameDateModified.columns

#Creating a dataframe of the negative amounts , We have 17 rows in this dataset, A common feature is that they have value '--' as pick up point
negatives = uberMergedNameDateModified[uberMergedNameDateModified['Transaction amount in KES (incl. taxes)'] < 0]
negatives.describe()
print(negatives.to_string())

#Further insights on rows with no definitive pick up point #we have some positive values without a pick up address
pickUpPoint = uberMergedNameDateModified[uberMergedNameDateModified['Pick-up address'] == "--"]
pickUpPoint.columns()
print(pickUpPoint.to_string())

#under the periodic billing we have 3 categories: Periodic Billing,VISA - 5566,VISA - 5558
#under transaction type we have 2 categories: Service & Technology Fee and Payment
pickUpPoint['Payment method'].value_counts()
pickUpPoint['Transaction type'].value_counts()

#dataset for positive values but with no pick up address
pickUpPointPositive = pickUpPoint[pickUpPoint['Transaction amount in KES (incl. taxes)'] > 0]
print(pickUpPointPositive.to_string())

#all the positive values with no pick up point are service and technology fees periodic billing
pickUpPointPositive['Payment method'].value_counts()
pickUpPointPositive['Transaction type'].value_counts()

#dataframe for values greater than zero
uberBills = uberMergedNameDateModified[uberMergedNameDateModified['Transaction amount in KES (incl. taxes)'] > 0]
print(uberBills.to_string())
uberBills.describe()

#Assingning a department to each employee
EmployeeList = pd.read_csv(r'D:\python\pycharm\uberAnalysis\Data\EmployeeNamesUpdated.csv')

#employee names
names = list(EmployeeList['Employee Name'])

#creating an empty list that we will add department names
department = []

#for loop to match department index from both datasets
for r in range(len(uberMergedNameDateModified['Fullname'])):
    if uberMergedNameDateModified['Fullname'][r] in names:
        idx1 = names.index(uberMergedNameDateModified['Fullname'][r])
        department.append(EmployeeList['Department'][(idx1)])

print((pd.DataFrame(department)).head())
uberMergedNameDateModified['Departments'] = department
print(uberMergedNameDateModified.to_string())

#uberNames = list(uberMergedNameDateModified['Fullname'])
#T = [names.index(uberMergedNameDateModified['Fullname'][i]) for i in uberNames]
#T2 = [uberNames.index(r) for r in names]
#print("is"+str(T))
#department.append(EmployeeList['Department'][T])
#print((pd.DataFrame(department)).head())

# Analysis of negative amounts
negatives.columns
negatives['Month']

# A total of KES 751,478 was paid out Vs KES 154,239 in 2022 for the period under consideration
totalsPerYearOutflows = negatives.groupby('Year')['Transaction amount in KES (incl. taxes)'].sum()

#totals for a comparative period of 2022 in 2021
FilteredDates2021 = negatives.loc[(negatives['Transaction timestamp (UTC)'] >='2021-01-01') & (negatives['Transaction timestamp (UTC)']<'2021-04-30')]
FilteredDates2022 = negatives.loc[(negatives['Transaction timestamp (UTC)'] >='2022-01-01') & (negatives['Transaction timestamp (UTC)']<'2022-04-30')]

#for the period Jan to April; KES 154,239 in 2022 was paid Vs KES 282,863 in 2021
sum2021 = FilteredDates2021.groupby("Year")['Transaction amount in KES (incl. taxes)'].sum()
sum2022 = FilteredDates2022.groupby("Year")['Transaction amount in KES (incl. taxes)'].sum()

#Bills paid analysis
uberBills = uberMergedNameDateModified[uberMergedNameDateModified['Transaction amount in KES (incl. taxes)'] > 0]
bills2021 = uberBills[uberBills['Year'] == 2021]
bills2022 = uberBills[uberBills['Year'] == 2022]

GroupBills2021 =bills2021.groupby('Month').sum()
GroupBills2022 = bills2022.groupby('Month').sum()

FilteredDates2021Bills = bills2021.loc[(bills2021['Transaction timestamp (UTC)'] >='2021-01-01') & (bills2021['Transaction timestamp (UTC)']<'2021-04-30')]
GroupBills2021Filtered = FilteredDates2021Bills.groupby("Month",).sum()
#2022 has comparatively higher spending on rides than 2021
x21 =['January', 'February', 'March', 'April']
y201 = GroupBills2021Filtered['Transaction amount in KES (incl. taxes)']
y202 = GroupBills2022['Transaction amount in KES (incl. taxes)']
X_axis2m = np.arange(len(x21))
plt.bar(X_axis2m - 0.2, y201,0.4,label = '2021', color = 'grey')
plt.bar(X_axis2m - 0.4, y202,0.4,label = '2022')
plt.xticks(X_axis2m, x21)
plt.xlabel("Mothly period")
plt.ylabel("Amount in KES")
plt.title("Expediture Per Month")
plt.legend()
plt.show()

#expenditure trend for 2021,towards the last quarter the expenditure is rising and this could be a continuous trend to Jan 2022
months =['January', 'February', 'March', 'April', 'May', 'June', 'july', 'August', 'September','October','November','December']
monthXaxis = np.arange(len(months))
y = GroupBills2021['Transaction amount in KES (incl. taxes)']
plt.bar(months, y , label = "2022")
plt.xlabel("Mothly period")
plt.ylabel("Amount in KES")
plt.title("Expediture Per Month 2021")
plt.show()

#A histogram to note the distribution of number of trips per month

l = bills2021['Month']
plt.hist(l,bins = [1,2,3,4,5,6,7,8,9 ,10 ,11 ,12, 13],color='grey' ,edgecolor='black')
#plt.hist(l , bins = [0 - 0.5,1,2,3,4,5,6,7,8,9 ,10 ,11 ,12] ,color='black' ,edgecolor='black')
plt.xlabel('Months')
plt.ylabel('No. of trips')
plt.title('Trip distribution per Month in 2021')
plt.show()

#Histogram to compare trip distribution .. Both the two plots...
plt.rcParams['figure.figsize'] = [7.5, 3.5]
plt.rcParams["figure.autolayout"] = True
g = pd.DataFrame(FilteredDates2021Bills['Month'])
h = pd.DataFrame(bills2022['Month'])
fig,axes = plt.subplots(1,2)
g.hist('Month',ax=axes[0],color='grey' ,edgecolor='black',)
h.hist('Month',ax=axes[1],color='brown' ,edgecolor='black')
#plt.xlabel('Months')
#plt.ylabel('No. of trips')
#plt.title('Trip distribution per Month')
plt.show()

#trips per month for Jan,Feb,March,April
x = FilteredDates2021Bills['Month']
plt.subplot(1,2,1)
plt.hist(x,bins = [1,2,3,4,5],color='grey' ,edgecolor='orange')
plt.title('Trip distribution per month 2021')

y = bills2022['Month']
plt.subplot(1,2,2)
plt.hist(y,bins = [1,2,3,4,5],color='brown' ,edgecolor='orange')
plt.title('Trip distribution per month 2022')

plt.suptitle("Trip Distribution Per Month (Jan,Feb,March,April)")
plt.show()

#Total number of trips in the period ...The number of trips in 2022 is less by
NumberOfTrips2022 = bills2022['Year'].value_counts()
NumberOfTrips2021Filtered = FilteredDates2021Bills['Year'].value_counts()

#Mean price per trip for 2021 and 2022 ... The average price in 2022 is higher by KES 152
MeanPrice2021 = FilteredDates2021Bills['Transaction amount in KES (incl. taxes)'].mean()
MeanPrice2022 = bills2022['Transaction amount in KES (incl. taxes)'].mean()
differenceInPricing = MeanPrice2021-MeanPrice2022
FilteredDates2021Bills.columns

#Total amounts for the spent in 2021 and 2022..2022 has higher bills even though it accounts for less number of trips
#in a comparable period
MeanPrice2021 = FilteredDates2021Bills['Transaction amount in KES (incl. taxes)'].sum()
MeanPrice2022 = bills2022['Transaction amount in KES (incl. taxes)'].sum()

#Total expediture per department
DepartmentExpenditure2021Filtered = FilteredDates2021Bills.groupby('Departments').sum()
DepartmentExpenditure2022 = bills2022.groupby('Departments').sum()





