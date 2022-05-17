import pandas as pd

uberAugApril = pd.read_csv(r'D:\python\pycharm\uberAnalysis\Data\uber_AugApril22.csv')
uberJanJuly = pd.read_csv(r'D:\python\pycharm\uberAnalysis\Data\uberJanJuly21.csv')

uberMerged = pd.concat([uberJanJuly,uberAugApril],ignore_index = True)
print(uberMerged.to_string())
uberMerged.columns

uberMerged.to_csv(r"D:\python\pycharm\uberAnalysis\Data\uberMerged.csv" , index = True)

uberMerged2 = uberMerged.drop(['Trip.Eats.ID', 'Transaction timestamp (UTC)', 'Request date (UTC)',
       'Request time (UTC)','Drop-off date (UTC)', 'Drop-off time (UTC)','Request timezone offset from UTC','Invoices',
       'Programme','Group','Fare in local currency (excl. taxes)', 'Taxes in local currency',
       'Tip in local currency','Transaction amount in local currency (incl. taxes)','Local currency code',
        'Fare in KES (excl. taxes)','Taxes in KES','Tip in KES'], axis=1)

uberMerged2.columns

print(uberMerged2.to_string())