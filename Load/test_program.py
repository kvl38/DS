import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('./Load/data/Netsim-Connection-Metrics-load-0a8e3a50-e006-4659-81ca-c8b43b628a39.txt')
df = pd.read_csv('./data/txt.csv')

# list = []
# for i in range(1,1063):
#     list.append(str(i))
# print (','. join(list))

# df["1"].plot() # 1 показывает нагрузку на 1-м сервере (от времени)
graf = np.mean(df.loc()[:])
print(graf)
# graf.plot() #средняя нагрузка каждого сервера

graf2 = np.mean(df[:], axis=1) #среднее значение от времени
# graf2.plot() #средняя нагрузка (от времени)

graf3 = df.mean()
graf3.plot() #средняя нагрузка на каждом сервере
# plt.show()

