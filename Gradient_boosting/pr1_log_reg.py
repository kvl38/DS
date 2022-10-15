import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def Solution(x_train, y_train, task_number, condition):
    start_time = datetime.datetime.now()
    clf = LogisticRegression()
    result = np.mean(cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc'))
    time = datetime.datetime.now() - start_time
    print(f'\n{task_number}. Качество логистической регресси {condition}:', str(result), '\nВремя выполнения: ' + str(time))

#1. Качество логистической регресси
df = pd.read_csv('features.csv', index_col='match_id')
data_to_drop = ['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                'barracks_status_radiant', 'barracks_status_dire']
X = df.drop(data_to_drop, axis=1)
y = df['radiant_win']
X = X.fillna(0)
X = StandardScaler().fit_transform(X)

Solution(X, y, 1, condition="")


# 2. Качество логистической регрессии без категориальных признаков
categorical_features = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero',
                        'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']

df.drop(data_to_drop, inplace=True, axis=1)
df.drop(categorical_features, inplace=True, axis=1)

X2 = df
X2 = StandardScaler().fit_transform(X2.fillna(0))

Solution(X2, y, 2, condition="без категориальных признаков")


#3. Количество идентификаторов героев в данной игре
dfh = pd.read_csv('./data/dictionaries/heroes.csv')
N = dfh.count()[0]
print("\n3. Количество идентификаторов героев в игре :", N)


#4. Качество при добавлении "мешка слов" по героям
# N — количество различных героев в выборке
data = pd.read_csv('features.csv', index_col='match_id')
X_pick = np.zeros((data.shape[0], N))

for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_pick = pd.DataFrame(X_pick)
X2 = pd.DataFrame(X2)
X3 = pd.concat([X2, X_pick], axis=1)
X3 = StandardScaler().fit_transform(X3.fillna(0))
Solution(X3, y, 4, condition="при добавлении \"мешка слов\" по героям")


#5. Минимальное и максимальное значение прогноза на тестовой выборке

X_test = pd.read_csv('features_test.csv', index_col='match_id')

X_pick2 = np.zeros((X_test.shape[0], N))

for i, match_id in enumerate(X_test.index):
    for p in range(5):
        X_pick2[i, X_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick2[i, X_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_test.drop(categorical_features, inplace=True, axis=1)
X_pick2 = pd.DataFrame(X_pick2)
X_test = pd.DataFrame(X_test)
X_test = pd.concat([X_test, X_pick2], axis=1)
X_test = StandardScaler().fit_transform(X_test.fillna(0))

X_train = X3
start_time = datetime.datetime.now()
clf = LogisticRegression()
clf.fit(X3, y)
y_pred = clf.predict_proba(X_test)[:, 1]
time = datetime.datetime.now() - start_time
print("5. Значение прогноза на тестовой выборке")
print('Минимальное значение прогноза:', min(y_pred))
print('Максимальное значение прогноза:', max(y_pred))
print("Время выполнения:", str(time))



