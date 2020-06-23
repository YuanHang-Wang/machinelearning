# code
import pandas as pd

voice_data=pd.read_csv('voice.csv')
x=voice_data.iloc[:,:-1]
y=voice_data.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=0, strategy='mean')
x=imp.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x_train)
x_train = scaler1.transform(x_train)
x_test = scaler1.transform(x_test)

from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression(max_iter=10000)
logistic.fit(x_train,y_train)

from sklearn.neural_network import MLPClassifier
nn=MLPClassifier(max_iter=100000)
nn.fit(x_train,y_train)

from sklearn.tree import DecisionTreeClassifier
cart=DecisionTreeClassifier()
cart.fit(x_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,criterion="gini")
rf.fit(x_train,y_train)

from sklearn.svm import SVC
svc = SVC(C=1,kernel='rbf', probability=True)
svc.fit(x_train, y_train)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

from sklearn import metrics

y_train_result = cart.predict(x_train)
print('cart train Accuracy Score:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = cart.predict(x_test)
print('cart test Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')

y_train_result = svc.predict(x_train)
print('svc train Accuracy Score:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = svc.predict(x_test)
print('svm test Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')

y_train_result = logistic.predict(x_train)
print('logistic train Accuracy Score:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = svc.predict(x_test)
print('logistic test Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')

y_train_result = knn.predict(x_train)
print('knn train Accuracy Score:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = knn.predict(x_test)
print('knn test Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')

y_train_result = nn.predict(x_train)
print('nn train Accuracy Score:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = nn.predict(x_test)
print('nn test Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('\n')

y_train_result = rf.predict(x_train)
print('rf train Accuracy Score:')
print(metrics.accuracy_score(y_train_result, y_train))
y_pred = rf.predict(x_test)
print('rf test Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
