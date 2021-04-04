# junk.py

import pandas as pd

def junk():
    data = pd.read_csv("Mall_Customers.csv")
    print(data.isnull().any()) # Check for null values

junk()