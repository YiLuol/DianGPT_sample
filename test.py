import pandas as pd
path="eval.xlsx"
data=pd.read_excel(path)
data_list = data.to_numpy().tolist()
index=len(data_list)
print(data_list,index)