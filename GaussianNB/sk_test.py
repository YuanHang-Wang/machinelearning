import pandas as pd
# noinspection PyUnresolvedReferences
import numpy as np
import time
time_start = time.time()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
if __name__ == '__main__':
    accuracy_rate1 = []
    accuracy_rate3 = []
    print('输入训练次数:')
    times = input()
    for i in range(int(times)):
        voice_data = pd.read_csv('voice.csv')
        x = voice_data.iloc[:, :-1]
        y = voice_data.iloc[:, -1]
        y=LabelEncoder().fit_transform(y)
        sim = SimpleImputer(missing_values=0,strategy= 'mean')
        x = sim.fit_transform(x)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

        std_scale = preprocessing.StandardScaler().fit(x_train)
        x_train_std = std_scale.transform(x_train)
        x_test_std = std_scale.transform(x_test)

        # on non-standardized data
        pca = PCA(n_components=11).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

        # om standardized data
        pca_std = PCA(n_components=11).fit(x_train_std)
        x_train_std = pca_std.transform(x_train_std)
        x_test_std = pca_std.transform(x_test_std)

        mnb = MultinomialNB()
        gnb = GaussianNB()
        """
         gnb.fit(x_train_std , y_train)
         y_predict = mnb.predict(x_test_std)
         print('accuracy:', mnb.score(x_test_std,y_test), classification_report(y_test, y_predict))
         """
        gnb.fit(x_train_std, y_train)
        y_predict = gnb.predict(x_test_std)
        print('accuracy:', gnb.score(x_test_std,y_test), classification_report(y_test, y_predict))
        #print('accuracy:', gnb.score(x_test_std,y_test))
        accuracy_rate = gnb.score(x_test_std, y_test)
        accuracy_rate1.append(accuracy_rate)
        gnb.fit(x_train, y_train)
        y_predict = gnb.predict(x_test)
        print('accuracy:', gnb.score(x_test, y_test), classification_report(y_test, y_predict))
        #print('accuracy:', gnb.score(x_test, y_test))
        accuracy_rate0 = gnb.score(x_test, y_test)
        accuracy_rate3.append(accuracy_rate0)
    accuracy_rate2 = np.array(accuracy_rate1)
    accuracy_rate4 = np.array(accuracy_rate3)
    print('accuracy_rate: %f' % (accuracy_rate2.mean()))
    print('accuracy_rate: %f' % (accuracy_rate4.mean()))
    x = range(1, int(times) + 1)
    plt.plot(x, accuracy_rate1)
    plt.plot(x, accuracy_rate3)
    plt.title('accuracy', fontsize=24)
    plt.xlabel('Times', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    ax = plt.gca()
    plt.ylim(0 , 1)
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()
    ac0=[]
    ac1=[]
    for i in range(1,21):
        voice_data = pd.read_csv('voice.csv')
        x = voice_data.iloc[:, :-1]
        y = voice_data.iloc[:, -1]
        y = LabelEncoder().fit_transform(y)
        sim = SimpleImputer(missing_values=0, strategy='mean')
        x = sim.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        std_scale = preprocessing.StandardScaler().fit(x_train)
        x_train_std = std_scale.transform(x_train)
        x_test_std = std_scale.transform(x_test)

        pca = PCA(n_components=int(i)).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)

        pca_std = PCA(n_components=int(i)).fit(x_train_std)
        x_train_std = pca_std.transform(x_train_std)
        x_test_std = pca_std.transform(x_test_std)

        gnb = GaussianNB()
        gnb.fit(x_train_std, y_train)
        y_predict = gnb.predict(x_test_std)

        accuracy_rate = gnb.score(x_test_std, y_test)
        ac0.append(accuracy_rate)
        gnb.fit(x_train, y_train)
        y_predict = gnb.predict(x_test)
        accuracy_rate0 = gnb.score(x_test, y_test)
        ac1.append(accuracy_rate0)
    x = range(1, 21)
    plt.plot(x, ac0)
    plt.plot(x, ac1)
    plt.title('accuracy', fontsize=24)
    plt.xlabel('dimension', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    ax = plt.gca()
    plt.ylim(0, 1)
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()
    time_end = time.time()
    print('time used: %fs' % (time_end - time_start))