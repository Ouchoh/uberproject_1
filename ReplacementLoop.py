import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

uber_1t = pd.read_csv(r"D:\python\pycharm\uberAnalysis\Data\uber_1718_newsT2.csv")

uber_1t["Expense Code"].unique()

# a loop to replace values in a column with a unique value
for x in range(len(uber_1t["Expense Code"])):

    if uber_1t["Expense Code"][x] in ['client support', 'support', 'Support', 'National bank kenyatta Avenue']:

        uber_1t["Expense Code"][x] = 'SUPPORT'

    elif uber_1t["Expense Code"][x] in ['meeting','Meeting', '1517']:

        uber_1t["Expense Code"][x] = 'MEETING'

    elif uber_1t["Expense Code"][x]  in ['work late', 'working late', 'WORK LATE']:

        uber_1t["Expense Code"][x] = 'WORKLATE'

    elif uber_1t["Expense Code"][x]  in ['nssf settlement', 'nssf', 'follow up on nssf', 'nssf follow up', 'from bank to office',

        'payment of PAYE and NHIF', 'nakumatt lifestyle']:

        uber_1t["Expense Code"][x] = 'OFFICE ADMIN'

    else:

        uber_1t["Expense Code"][x]


