# This code is prepared for the knn method

import numpy as np
import random
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from scipy import sparse
import heapq
import math
import sys


def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    tem_data = tem[:, 1:-1]
    tem_label = tem[:, -1]
    data = tem_data.astype(np.float)
    label = tem_label.astype(np.float).astype(np.int)
    return data, label

def load_npz_data(file_name):
    data = sparse.load_npz(file_name)
    data = data.toarray()
    return data



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
        new_train_data = scaler.transform(train_data)
        pca = condense_data_pca(new_train_data, num_of_components)
        new_train_data = pca.transform(new_train_data)
        new_data = scaler.transform(Data)
        new_data = pca.transform(new_data)
    elif kernelpca_or_not :
        new_train_data = scaler.transform(train_data)
        kernelpca = condense_data_kernel_pca(new_train_data, num_of_components)
        new_train_data = kernelpca.transform(new_train_data)
        new_data = scaler.transform(Data)
        new_data = kernelpca.transform(new_data)
    else:
        new_train_data = scaler.transform(train_data)
        new_data = scaler.transform(Data)
    return new_data, new_train_data

def exchange(test_y):
    ex_ty_list = []
    rank_ty = []
    for i in range(len(test_y)):
        ex_ty_list.append((int(test_y[i]),i+1))
    exed_ty = sorted(ex_ty_list)
    for i in exed_ty:
        rank_ty.append(i[1])
    return rank_ty


def extract_data_from_dataset(dataset, index):
    tem = []
    for each in index:
        tem.append(dataset[each])
    tem = np.array(tem)
    return tem


def generate_train_data(Data, Label, primal_label, target_rate, knn_list, knn_label_list, final_knn_list, distance_matric):
    num_of_minority = count_num(primal_label, 1)
    num_of_nn = len(final_knn_list)
    rate = num_of_nn / num_of_minority
    resampling_nn(rate, target_rate, num_of_nn, num_of_minority, knn_list, knn_label_list, final_knn_list, distance_matric)
    positive_samples = np.where(primal_label == 1)
    positive_samples = positive_samples[0].tolist()
    final_knn_list += positive_samples
    random.shuffle(final_knn_list)
    train_x = extract_data_from_dataset(Data, final_knn_list)
    train_y = extract_data_from_dataset(Label, final_knn_list)
    return train_x,train_y

# def generate_test_data(final_knn_list, Data, Label):
#     tem_data_test = []
#     tem_label_test = []
#     num_of_rows = Data.shape[0]
#     for group_index_start in range(num_of_rows):
#         if group_index_start not in final_knn_list:
#             tem_data_test += Data[group_index_start, :]
#             tem_label_test += Label[group_index_start, :]
#     test_data = np.array(tem_data_test)
#     test_label = np.array(tem_label_test)
#     return test_data, test_label

def generate_test_data(Data, Label, Ds, num_of_train):
    test_data = Data[Ds[num_of_train]:, :]
    test_label = Label[Ds[num_of_train]:]
    return test_data, test_label




def count_num(data, target):
    if not isinstance(data, list):
        t = data.tolist()
    else:
        t = data
    num = t.count(target)
    return num

def find_rate(data):
    zeor_num = count_num(data, 0)
    one_num = count_num(data, 1)
    rate = zeor_num / one_num
    return rate


def distance(v1,v2):
    return np.linalg.norm(v1-v2)

def create_distance_matric(data):
    num_of_rows = data.shape[0]
    distance_matric = np.zeros((num_of_rows,num_of_rows))
    for item in range(num_of_rows):
        if item % 100 == 0:
            print(item)
        for num in range(item,num_of_rows):
            d = distance(data[item],data[num])
            distance_matric[item,num] = d
            distance_matric[num,item] = d
    return distance_matric

def create_knn_list(distance_matric, k):
    knn = []
    for each in distance_matric:
        knn.append(heapq.nsmallest(k, range(len(each)), each.take))
    return knn

def transform_knn_list_to_label(knn_list, label):
    num_of_samples = len(knn_list)
    knn_label_list = []
    for pos in range(num_of_samples):
        tem_label = []
        for each in knn_list[pos]:
            if label[each] == 0:
                tem_label.append(0)
            else:
                tem_label.append(1)
        knn_label_list.append(tem_label)
    return knn_label_list

def eliminate_noisy(knn_label_list, selected_threshold, noise_threshold, label):
    num_of_samples = len(knn_label_list)
    final_knn_list = []
    for pos in range(num_of_samples):
        if label[pos] == 0:
            one_num = count_num(knn_label_list[pos], 1)
            if one_num > selected_threshold and one_num < noise_threshold:
                final_knn_list.append(pos)
    return final_knn_list


def oversampling_nn(rate, target_rate, num_of_nn, num_of_minority, knn_list, knn_label_list, final_knn_list, distance_matric):
    while rate < target_rate:
        index = random.randint(0, num_of_nn-1)
        # print(index)
        index = final_knn_list[index]
        # print(index)
        label = []
        # print(knn_label_list[index])
        for each in range(len(knn_label_list[index])):
            if knn_label_list[index][each] == 0:
                label.append(each)
        # print(label)
        # print(knn_list[index])
        nn = knn_list[index].take(label)
        # print(nn)
        # print(type(nn))
        nn = nn.tolist()
        distance = distance_matric[index].take(nn)
        # print(distance)
        i = 1
        while i <= len(label):
            nn_index = heapq.nsmallest(i, range(len(distance)), distance.take)
            # print(type(nn_index))
            # print(nn_index)
            # print(nn_index[-1])
            if nn[nn_index[-1]] not in final_knn_list:
                final_knn_list.append(nn[nn_index[-1]])
                break
            i += 1
        rate = len(final_knn_list)/num_of_minority
        # print(rate)


# def undersamling_nn(rate, target_rate, num_of_nn, num_of_minority, knn_list, knn_label_list, final_knn_list, distance_matric):
#     while rate > target_rate:
#         index = random.randint(0, len(final_knn_list)-1)
#         del final_knn_list[index]
#         rate = len(final_knn_list) / num_of_minority
#         # print(rate)


def undersamling_nn(rate, target_rate, num_of_nn, num_of_minority, knn_list, knn_label_list, final_knn_list, distance_matric):
    num = math.ceil(num_of_minority * target_rate)
    final_knn_distance = []
    for index in final_knn_list:
        label = []
        # print(knn_label_list[index])
        for each in range(len(knn_label_list[index])):
            if knn_label_list[index][each] == 1:
                label.append(each)
        nn = knn_list[index].take(label)
        nn = nn.tolist()
        distance = distance_matric[index].take(nn)
        nn_index = heapq.nsmallest(1, range(len(distance)), distance.take)
        final_knn_distance.append(distance[nn_index[0]])
    final_knn_distance = np.array(final_knn_distance)
    the_index = heapq.nsmallest(num, range(len(final_knn_distance)), final_knn_distance.take)
    final_knn_list = np.array(final_knn_list)
    final_knn_list = final_knn_list.take(the_index)
    final_knn_list = final_knn_list.tolist()




def resampling_nn(rate, target_rate, num_of_nn, num_of_minority, knn_list, knn_label_list, final_knn_list, distance_matric):
    if rate < 1.5:
        oversampling_nn(rate, target_rate, num_of_nn, num_of_minority, knn_list, knn_label_list, final_knn_list, distance_matric)
    else:
        undersamling_nn(rate, target_rate, num_of_nn, num_of_minority, knn_list, knn_label_list, final_knn_list, distance_matric)




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
    agr = []
    agtea = []
    aga = []
    tt = []
    rt = []
    return gp,gr,ga,agtp,agr,agtea,aga,tt,rt

def aver(l):
    if len(l) == 0:
        return 0
    else:
        return sum(l)/len(l)

def scan_file(file_name):
    f = open(file_name,'r')
    gp,gr,ga,agtp,agr,agtea,aga,tt,rt = initlist()
    for i in f:
        word,num = divide_alpha_digit(i)
        if word == 'the average group top precision is ':
            agtp.append(num)
        if word == 'the average group recall is ':
            agr.append(num)
        if word == 'the average group top exact accuracy is ':
            agtea.append(num)
        if word == 'the average group accuracy is ':
            aga.append(num)
        if word == 'the  time training time is ':
            tt.append(float(str(num)[1:-1]))
        if word == 'the  time running time is ':
            rt.append(float(str(num)[1:-1]))
    av_aptp = aver(agtp)
    av_agr = aver(agr)
    av_agtea = aver(agtea)
    av_aga = aver(aga)
    av_tt = aver(tt)
    av_rt = aver(rt)
    return av_aptp, av_agr, av_agtea,av_aga,av_tt,av_rt

def append_file(file_name):
    av_agtp, av_agr, av_agtea, av_aga, av_tt, av_rt = scan_file(file_name)
    fscore = (2 * av_agtp * av_agr) / (av_agtp + av_agr)
    f = open(file_name,'a')
    f.write("the F-score is {0}\n".format(fscore))
    f.write("the average group top precision is {0}\n".format(av_agtp))
    f.write("the average group recall is {0}\n".format(av_agr))
    f.write("the average group top exact accuracy is {0}\n".format(av_agtea))
    f.write("the average group accuracy is {0}\n".format(av_aga))
    f.write("the 3 time training time is {0}\n".format(av_tt))
    f.write("the 3 time running time is {0}\n".format(av_rt))
    f.close()



