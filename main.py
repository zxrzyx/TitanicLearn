import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier

n_trains_data = pd.read_csv('all/train.csv')
n_test_data = pd.read_csv('all/test.csv')
# print(n_trains_data[:9])
#
survived_1 = n_trains_data.Pclass[n_trains_data.Survived == 1].value_counts()
survived_0 = n_trains_data.Pclass[n_trains_data.Survived == 0].value_counts()
n_survived_label = n_trains_data['Survived']

cabin_1 = n_trains_data.Survived[pd.notnull(n_trains_data.Cabin)].value_counts()
cabin_0 = n_trains_data.Survived[pd.isnull(n_trains_data.Cabin)].value_counts()

sv = pd.DataFrame({'None': cabin_0, 'Survived': cabin_1}).transpose()
# sv.plot(kind='bar')
# plt.show()

n_trains_data['Sex'] = n_trains_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
n_trains_data['Embarked'] = n_trains_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, None: 3}).astype(int)
n_trains_data['Age'] = n_trains_data['Age'].fillna(n_trains_data['Age'].median())


n_trains_data.loc[(n_trains_data['Age'] <= 16.336), 'Age'] = 0
n_trains_data.loc[(n_trains_data['Age'] > 16.337) & (n_trains_data['Age'] <= 32.252), 'Age'] = 1
n_trains_data.loc[(n_trains_data['Age'] > 32.253) & (n_trains_data['Age'] <= 48.168), 'Age'] = 2
n_trains_data.loc[(n_trains_data['Age'] > 48.169) & (n_trains_data['Age'] <= 64.084), 'Age'] = 3
n_trains_data.loc[(n_trains_data['Age'] > 64.085) & (n_trains_data['Age'] <= 80), 'Age'] = 4

n_trains_data['Fare'] = n_trains_data['Fare'].fillna(n_trains_data['Fare'].median())
n_trains_data['Fare'] = scale(n_trains_data['Fare'])

b = a = pd.cut(n_trains_data['Fare'], 5)
print(b)

n_trains_data.loc[(n_trains_data['Fare'] >= -0.659) & (n_trains_data['Fare'] <= 1.145), 'Fare'] = 0
n_trains_data.loc[(n_trains_data['Fare'] > 1.145) & (n_trains_data['Fare'] <= 3.478), 'Fare'] = 1
n_trains_data.loc[(n_trains_data['Fare'] > 3.478) & (n_trains_data['Fare'] <= 5.541), 'Fare'] = 2
n_trains_data.loc[(n_trains_data['Fare'] > 5.541) & (n_trains_data['Fare'] <= 7.604), 'Fare'] = 3
n_trains_data.loc[(n_trains_data['Fare'] > 7.604) & (n_trains_data['Fare'] <= 9.667), 'Fare'] = 4

n_trains_data.loc[(n_trains_data.Cabin.notnull()), 'Cabin'] = 1
n_trains_data['Cabin'] = n_trains_data['Cabin'].fillna(0)


n_test_data['Sex'] = n_test_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
n_test_data['Embarked'] = n_test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, None: 3}).astype(int)
n_test_data['Age'] = n_test_data['Age'].fillna(n_trains_data['Age'].median())

n_test_data.loc[(n_test_data['Age'] <= 16.336), 'Age'] = 0
n_test_data.loc[(n_test_data['Age'] > 16.337) & (n_test_data['Age'] <= 32.252), 'Age'] = 1
n_test_data.loc[(n_test_data['Age'] > 32.253) & (n_test_data['Age'] <= 48.168), 'Age'] = 2
n_test_data.loc[(n_test_data['Age'] > 48.169) & (n_test_data['Age'] <= 64.084), 'Age'] = 3
n_test_data.loc[(n_test_data['Age'] > 64.085) & (n_test_data['Age'] <= 80), 'Age'] = 4

n_test_data['Fare'] = n_test_data['Fare'].fillna(n_test_data['Fare'].median())
n_test_data['Fare'] = scale(n_test_data['Fare'])
n_test_data.loc[(n_test_data['Fare'] >= -0.659) & (n_test_data['Fare'] <= 1.145), 'Fare'] = 0
n_test_data.loc[(n_test_data['Fare'] > 1.145) & (n_test_data['Fare'] <= 3.478), 'Fare'] = 1
n_test_data.loc[(n_test_data['Fare'] > 3.478) & (n_test_data['Fare'] <= 5.541), 'Fare'] = 2
n_test_data.loc[(n_test_data['Fare'] > 5.541) & (n_test_data['Fare'] <= 7.604), 'Fare'] = 3
n_test_data.loc[(n_test_data['Fare'] > 7.604) & (n_test_data['Fare'] <= 9.668), 'Fare'] = 4

n_test_data.loc[(n_test_data.Cabin.notnull()), 'Cabin'] = 1
n_test_data['Cabin'] = n_test_data['Cabin'].fillna(0)

drop_ele = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Survived']
drop_ele_2 = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch']

n_trains_data.to_csv('clean_train.csv')

n_trains = n_trains_data.drop(drop_ele, axis=1)
n_test = n_test_data.drop(drop_ele_2, axis=1)

print(n_trains[:9])
param_range = np.logspace(-6, 0.5, 5)


X_trains, X_test, y_trains, y_test = train_test_split(n_trains, n_survived_label, test_size=0.3)

classifier = SVC(gamma=3)
classifier.fit(X_trains, y_trains)
# classifier.fit(n_trains, n_survived_label)
train_loss, test_loss = validation_curve(SVC(), n_trains, n_survived_label,
                                         param_name='gamma', param_range=param_range, cv=10,
                 scoring='neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
plt.plot(param_range, train_loss_mean, 'o-', color='g', label='Training')
plt.plot(param_range, test_loss_mean, 'o-', color='r', label='Cross-validation')
plt.xlabel('gamma')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

print(classifier.score(X_test, y_test))


result = classifier.predict(n_test)

result = np.array(result).reshape(-1, 1)

result_a = n_test_data['PassengerId']
result_a = np.array(result_a).reshape(-1, 1)

result_b = np.hstack([result_a, result])

result_0 = pd.DataFrame(result_b, columns=["PassengerId", "Survived"])
result_0.reset_index(drop=True)

# result_1 = pd.DataFrame(result)
# result_final = pd.concat([result_0, result_1])
result_0.to_csv('result.csv')
# print(result_0)


