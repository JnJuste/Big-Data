import pandas as pd

#Replacing the empty cells with 11 where on the Grade Axis 
df = pd.read_excel('Dset.xlsx')
df["GRADE"].fillna(11, inplace = True)
print(df.to_string())
