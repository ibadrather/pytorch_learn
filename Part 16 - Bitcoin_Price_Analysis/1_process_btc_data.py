import imp
import pandas as pd
import numpy as numpy
import time
import datetime
import os
from tqdm import tqdm

os.system("clear")

# Loading Data
data_path = r"bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
df = pd.read_csv(data_path)

# Converting timestamp column to date data
df.timestamp = df.timestamp.apply(
            lambda d: datetime.datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d')
            )

print(df)

# Groupping data by each day because I am interested in daily prices analysis.
# I will then perform operations on daily data
grouped_df = df.groupby(df.timestamp)

# Getting unique timestamp from dataframe. I will need them to access each group.
group_names = df.timestamp.unique()

# Total number of Groups or Days of data
print("Total length of Data is {} days.".format(len(group_names)))


# Now let us anaylse each day of data and see what would be appropriate next steps
for group_name in tqdm(group_names):
    group_df = grouped_df.get_group(group_name)
    group_df = group_df.dropna()
    print(group_df)
    break
