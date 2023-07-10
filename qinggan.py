# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdate
# from matplotlib.pyplot import rcParams 
# import time

data=pd.read_csv("1.csv",header=None,names=["发布时间","中心句","情感分数"])
timedata = data['发布时间'].copy()
time = []
for i in range(len(data)):
    timedata.loc[i]=timedata.loc[i][:10]
data['发布时间']=timedata
# Convert the date to datetime64
timedata = pd.to_datetime(timedata, format='%Y-%m-%d')
data1 = data['情感分数'].groupby(timedata).mean()
print(data1)
