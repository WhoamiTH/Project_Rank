import numpy as np
import random
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import heapq


def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    tem_data = tem[:, 1:-1]
    tem_label = tem[:, -1]
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


def standarize_PCA_data(Data):
    scaler = standardize_data(Data)
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
    return train_index_start,train_x,train_y

def generate_all_data(Ds, Dl, Data, Label, train_index_start, num_of_train, mirror_type, positive_value, negative_value):
    tem_data_train = []
    tem_label_train = []
    tem_data_test = []
    tem_label_test = []
    for group_index_start in range(len(Ds)):
        group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        temd = Data[group_start:group_start+length, :]
        teml = Label[group_start:group_start+length, :]
        if group_index_start<train_index_start + num_of_train:
            tem_data_train += temd
            tem_label_train += teml
        else:
            tem_data_test += temd
            tem_label_test += teml
    train_data = np.array(tem_data_train)
    train_label = np.array(tem_label_train)
    test_data = np.array(tem_data_test)
    test_label = np.array(tem_label_test)
    return train_data, train_label, test_data, test_label


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
        knn.append(heapq.nlargest(k, range(len(each)), each.take))
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



