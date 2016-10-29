import numpy as np
import pandas as pd

A = np.array([[1,2,3],[4,5,6]])


userp = pd.read_csv("interactions.csv", sep='\t')
pd.set_option('display.max_columns', 30)
print (userp.shape)
print (userp.head(50))

print('\n')
print(userp["user_id"].value_counts())
pd.to_datetime(userp["created_at"])
