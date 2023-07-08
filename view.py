import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.pyplot import rcParams 

# 防止中文乱码
rcParams['font.sans-serif'] = 'kaiti'
 
# 生成一个时间序列 
time =pd.to_datetime(np.arange(0,11), unit='D',
                   origin=pd.Timestamp('2023-07-01'))
 
# 生成数据
data = np.random.randint(0,1,size=11)
# 创建一个画布
fig = plt.figure(figsize=(12,9))
# 在画布上添加一个子视图
ax = plt.subplot(111)
# 这里很重要  需要将x轴的刻度 进行格式化
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
# 为X轴添加刻度
plt.xticks(pd.date_range(time[0],time[-1],freq='D'),rotation=45)
# 画折线
ax.plot(time,data,color='r')
# 设置标题
ax.set_title('折线图示例')
# 设置 x y 轴名称
ax.set_xlabel('日期',fontsize=20)
ax.set_ylabel('情感',fontsize=20)
plt.show()