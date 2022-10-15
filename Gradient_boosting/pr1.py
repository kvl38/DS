import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

# Подход 1: градиентный бустинг "в лоб"

data = pd.read_csv('features.csv', index_col='match_id')
test = pd.read_csv('features_test.csv', index_col='match_id')

#Целевая переменная
y_train = data['radiant_win']

print("\nЦелевая переменная: ['radiant_win']")
data.drop(["duration", "radiant_win", "tower_status_radiant", "tower_status_dire", "barracks_status_radiant", "barracks_status_dire"], axis=1, inplace=True)

#Пропуски в данных
print("\nПропуски в данных:\n", data.columns[data.isna().any()].tolist(), "\n\nКоличество пропусков:")

for i in data:
    sum = data['start_time'].count() - data[i].count()
    if sum != 0: print(i, ": ", sum)

x_train = data.fillna(0)
x_test = test.fillna(0)

kf = KFold(n_splits=5, shuffle=True)
number_of_trees = [10, 15, 30, 35, 40, 50]
for i in number_of_trees:
    start_time = datetime.datetime.now()
    print('\nКол-во деревьев: ', i)
    model = GradientBoostingClassifier(n_estimators=i, max_depth=2)
    model_scores = np.maen(cross_val_score(model, x_train, y_train, cv=kf, scoring='roc_auc'))
    print(model_scores)
    print('Время выполнения:', datetime.datetime.now() - start_time)