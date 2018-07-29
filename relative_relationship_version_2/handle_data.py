# This code for the relative_relationship_version_2

import numpy as np
import random
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    tem_data = tem[:, 1:-1]
    tem_label = tem[:, -1]
    # data = tem_data.astype(np.float).astype(np.int)
    data = tem_data.astype(np.float)
    label = tem_label.astype(np.float).astype(np.int)
    return data, label


def group(Data):
    i = 0
    j = 1
    l = 1
    t = Data[0][2]
    Ds = {}
    Dl = {}
    Ds[0] = 0
    while j < len(Data):
        if (t != Data[j][2]):
            Dl[i] = l
            Ds[i + 1] = j
            i += 1
            l = 1
            t = Data[j][2]
            j += 1
        elif (j == len(Data) - 1):
            Dl[i] = l + 1
            j += 1
        else:
            l += 1
            j += 1
    return Ds, Dl

def data_extend(Data_1, Data_2):
    m = list(Data_1)
    n = list(Data_2)
    return m + n

def condense_data_pca(Data, num_of_components):
    pca = PCA(n_components=num_of_components)
    pca.fit(Data)
    return pca


def condense_data_kernel_pca(Data, num_of_components):
    kernelpca = KernelPCA(n_components=num_of_components)
    kernelpca.fit(Data)
    return kernelpca


def standardize_data(Data):
    scaler = skpre.StandardScaler()
    scaler.fit(Data)
    return scaler


def standarize_PCA_data(train_data, Data, pca_or_not, kernelpca_or_not, num_of_components):
    scaler = standardize_data(train_data)
    if pca_or_not :
        new_data = scaler.transform(train_data)
        pca = condense_data_pca(new_data, num_of_components)
        new_data = scaler.transform(Data)
        new_data = pca.transform(new_data)
    elif kernelpca_or_not :
        new_data = scaler.transform(train_data)
        kernelpca = condense_data_kernel_pca(new_data, num_of_components)
        new_data = scaler.transform(Data)
        new_data = kernelpca.transform(new_data)
    else:
        new_data = scaler.transform(Data)
    return new_data

def exchange(test_y):
    ex_ty_list = []
    rank_ty = []
    for i in range(len(test_y)):
        ex_ty_list.append((int(test_y[i]),i+1))
    exed_ty = sorted(ex_ty_list)
    for i in exed_ty:
        rank_ty.append(i[1])
    return rank_ty


def generate_primal_train_data(Data,Label,Ds,Dl,num_of_train):
    train_index_start = 0
    front = Ds[train_index_start]
    end = Ds[train_index_start+num_of_train-1]+Dl[train_index_start+num_of_train-1]
    train_x = Data[front:end,:]
    train_y = Label[front:end]
    return train_x,train_y


def handleData_extend_mirror(Data, Label, start, length, positive_value, negative_value):
    temd = []
    teml = []
    for j in range(length):
        for t in range(length):
            if j != t:
                if 
                temd.append(data_extend(Data[start + j], Data[start + t]))
                if Label[start + j] > Label[start + t]:
                    # teml.append([-1])
                    teml.append([negative_value])
                else:
                    teml.append([positive_value])
    return temd, teml


def handleData_extend_not_mirror(Data, Label, start, length, positive_value, negative_value):
    temd = []
    teml = []
    for j in range(length):
        for t in range(j+1,length):
            temd.append(data_extend(Data[start + j], Data[start + t]))
            if Label[start + j] > Label[start + t]:
                teml.append([negative_value])
            else:
                teml.append([positive_value])
    return temd, teml


def generate_all_data(Ds, Dl, Data, Label, train_index_start, num_of_train, mirror_type, positive_value, negative_value):
    tem_data_train = []
    tem_label_train = []
    tem_data_test = []
    tem_label_test = []
    for group_index_start in range(len(Ds)):
        group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        if group_index_start< num_of_train:
            if mirror_type == 'mirror':
                temd, teml = handleData_extend_mirror(Data, Label, group_start, length, positive_value, negative_value)
            else:
                temd, teml = handleData_extend_not_mirror(Data, Label, group_start, length, positive_value, negative_value)
            tem_data_train = tem_data_train + temd
            tem_label_train = tem_label_train + teml
        else:
            temd, teml = handleData_extend_mirror(Data, Label, group_start, length, positive_value, negative_value)
            tem_data_test = tem_data_test + temd
            tem_label_test = tem_label_test + teml
    train_data = np.array(tem_data_train)
    train_label = np.array(tem_label_train)
    test_data = np.array(tem_data_test)
    test_label = np.array(tem_label_test)
    return train_data, train_label, test_data, test_label




def digit(x):
    if str.isdigit(x) or x == '.':
        return True
    else:
        return False

def alpha(x):
    if str.isalpha(x) or x == ' ':
        return True
    else:
        return False

def point(x):
    return x == '.'

def divide_digit(x):
    d = filter(digit, x)
    item = ''
    for i in d:
        item += i
    if len(item) == 0:
        return 0.0
    else:
        p = filter(point, item)
        itemp = ''
        for i in p:
            itemp += i
        # print(itemp)
        if len(itemp) > 1:
            return 0.0
        else:
            return float(item)

def divide_alpha(x):
    a = filter(alpha, x)
    item = ''
    for i in a:
        item += i
    return item

def divide_alpha_digit(x):
    num = divide_digit(x)
    word = divide_alpha(x)
    return word,num

def initlist():
    gp = []
    gr = []
    ga = []
    agtp = []
    agtea = []
    aga = []
    tt = []
    rt = []
    return gp,gr,ga,agtp,agtea,aga,tt,rt

def aver(l):
    return sum(l)/len(l)

def scan_file(file_name):
    f = open(file_name,'r')
    gp,gr,ga,agtp,agtea,aga,tt,rt = initlist()
    for i in f:
        word,num = divide_alpha_digit(i)
        if word == 'the average group top precision is ':
            agtp.append(num)
        if word == 'the average group top exact accuracy is ':
            agtea.append(num)
        if word == 'the average group accuracy is ':
            aga.append(num)
        if word == 'the  time training time is ':
            tt.append(float(str(num)[1:-1]))
        if word == 'the  time running time is ':
            rt.append(float(str(num)[1:-1]))
    av_aptp = aver(agtp)
    av_agtea = aver(agtea)
    av_aga = aver(aga)
    av_tt = aver(tt)
    av_rt = aver(rt)
    return av_aptp,av_agtea,av_aga,av_tt,av_rt

def append_file(file_name):
    av_aptp, av_agtea, av_aga, av_tt, av_rt = scan_file(file_name)
    f = open(file_name,'a')
    f.write("the average group top precision is {0}\n".format(av_aptp))
    f.write("the average group top exact accuracy is {0}\n".format(av_agtea))
    f.write("the average group accuracy is {0}\n".format(av_aga))
    f.write("the 3 time training time is {0}\n".format(av_tt))
    f.write("the 3 time running time is {0}\n".format(av_rt))
    f.close()



