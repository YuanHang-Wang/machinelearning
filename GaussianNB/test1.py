
import numpy as np
import pandas as pd
import random
import collections
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import itertools
import time
from sklearn.preprocessing import StandardScaler

def data(file_name):
    voice_data = pd.read_csv(file_name)
    x = voice_data.values[:, :-1]
    y = voice_data.values[:, -1]
    """
    train_num = random.sample(range(0, 3167), 2218)  # 设置随机数生成从0-3167中随机挑选2218个随机数
    test_num = list(set(range(0, 3167)).difference(set(train_num)))

    train_data = np.array(x)[train_num]
    train_tag = np.array(y)[train_num]

    test_data = np.array(x)[test_num]
    test_tag = np.array(y)[test_num]
    """
   #'''
#分层取样
    #男
    male_train_num = random.sample(range(0, 1583), 1109)
    male_test_num = list(set(range(0, 1583)).difference(set(male_train_num)))
    #女
    female_train_num = random.sample(range(1584, 3167), 1109)
    female_test_num = list(set(range(1584, 3167)).difference(set(female_train_num)))

    train_data = np.append(np.array(x)[male_train_num], np.array(x)[female_train_num], axis=0)#训练集数据
    train_tag = np.append(np.array(y)[male_train_num], np.array(y)[female_train_num], axis=0)#训练集标签
    test_data = np.append(np.array(x)[male_test_num], np.array(x)[female_test_num], axis=0)#测试集数据
    test_tag = np.append(np.array(y)[male_test_num], np.array(y)[female_test_num], axis=0)#测试集标签
   #'''
    _data = []
    _data1 = []
    _test1 = []
    _test0 = []
    for i in range(len(train_data)):
        _data.append(train_data[i])
    for i in range(len(test_data)):
        _data.append(test_data[i])
    for i in range(len(_data)):
        _test1.append(_data[i][4])
    piant(_test1)

    train_data = np.array(train_data)
    train_data *= 20
    test_data = np.array(test_data)
    test_data *= 20
    print(test_data[5][4])
    for i in range(len(train_data)):
        _data1.append(train_data[i])
    for i in range(len(test_data)):
        _data1.append(test_data[i])
    for i in range(len(_data1)):
        _test0.append(_data1[i][4])
    piant(_test0)
    return train_data, train_tag, test_data, test_tag

def piant(_test):
    x = range(0,3166)
    plt.plot(x, _test)
    plt.title('data', fontsize=24)
    plt.xlabel('number', fontsize=14)
    plt.ylabel('data', fontsize=14)
    plt.show()

def get(train_data, train_tag):
    male_list = []  # 男声的序号，列表
    female_list = []  # 女声的序号，列表
    for i in range(len(train_tag)):
        if train_tag[i] == 'male':
            male_list.append(i)
        else:
            female_list.append(i)
    continuousPara = {}  #创建字典
    for i in range(20): #特征值
        fea_data = train_data[male_list, i]
        mean = fea_data.mean()
        std = fea_data.std()
        continuousPara[(i, 'male')] = (mean, std)
        fea_data = train_data[female_list, i]
        mean = fea_data.mean()
        std = fea_data.std()
        continuousPara[(i, 'female')] = (mean, std)
    return continuousPara



# 计算P(feature = x|C)
def P_continuous(feature_Index, x, C, cP):
    fea_para = cP[(feature_Index, C)]
    mean = fea_para[0]
    std = fea_para[1]
    ans = 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(x - mean) ** 2) / (2 * std * std)) #连续属性概率预测
    return ans

def paintConfusion_float(lmr_matrix,classes):
    plt.figure(figsize = (15, 10))
    plt.imshow(lmr_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 90, size = 18)
    plt.yticks(tick_marks, classes, size = 18)
    plt.xlabel('Predict label', size = 20)
    plt.ylabel('True label', size = 20)
    lmr_matrix = lmr_matrix.astype('float') / lmr_matrix.sum(axis = 1)[:,np.newaxis]
    fmt='.6f'
    thresh = lmr_matrix.max() / 2.
    for i, j in itertools.product(range(lmr_matrix.shape[0]), range(lmr_matrix.shape[1])):
        plt.text(j, i, format(lmr_matrix[i, j], fmt),
                     horizontalalignment = "center",
                     color = "red" if lmr_matrix[i, j] > thresh else "black", size = 22)
    plt.tight_layout()
    plt.show()

# 高斯贝叶斯过程
def Bayes(test_data, train_tag, cP, times):
    # 求先验概率
    num = collections.Counter(train_tag)   #计数
    num = dict(num)        #创建male，female字典
    male_para = num['male'] / len(train_tag)
    female_para = num['female'] / len(train_tag)       #male，female 各自求比例

    Result = []
    for i in range(len(test_data)):
        ans_male = math.log(male_para)
        ans_female = math.log(female_para)
        for j in range(len(test_data[i])):
            #if j == 7|6|19:
                #continue
            #if j == 12:
                ans_male += math.log(P_continuous(j, test_data[i][j], 'male', cP))
                ans_female += math.log(P_continuous(j, test_data[i][j], 'female', cP))
        if ans_male > ans_female:
            Result.append('male')
        else:
            Result.append('female')
    return Result

if __name__ == '__main__':
    accuracy_rate1 = []
    accuracy_male1 = []
    accuracy_female1 = []
    print('输入训练次数:')
    times = input()
    time_start = time.time()
    for i in range(int(times)):
        train_data, train_tag, test_data, test_tag = data('voice.csv')  # 加载数据集
        continuousPara = get(train_data, train_tag)  # 求高斯分布需要的参数并构建字典
        predict_label = Bayes(test_data, train_tag, continuousPara, times)  # 高斯贝叶斯过程
        confusionMatrix = confusion_matrix(test_tag, predict_label, labels=['male', 'female'])  # 得出混淆矩阵
        classes = ['male', 'female']
        paintConfusion_float(confusionMatrix, classes)  # 绘制混淆矩阵，显示男女声的正确率与错误率
        print(classification_report(test_tag, predict_label))
        accuracy_male = confusionMatrix[0][0] / (len(test_tag) / 2)
        accuracy_female = confusionMatrix[1][1] / (len(test_tag) / 2)
        accuracy_rate = (confusionMatrix[0][0] + confusionMatrix[1][1]) / len(test_tag)
        classes = ['male', 'female']
        print('accuracy_rate: %f' % accuracy_rate)
        accuracy_rate1.append(accuracy_rate)
        accuracy_male1.append(accuracy_male)
        accuracy_female1.append(accuracy_female)
    accuracy_rate2 = np.array(accuracy_rate1)
    print('accuracy_rate: %f' % (accuracy_rate2.mean()))
    x = range(1,int(times)+1)
    plt.plot(x, accuracy_rate1)
    plt.plot(x, accuracy_male1)
    plt.plot(x, accuracy_female1)
    plt.title('accuracy', fontsize=24)
    plt.xlabel('Times', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    ax = plt.gca()
    plt.ylim(0.5, 1)
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

    time_end = time.time()
    print('Time used: %fs' % (time_end - time_start))