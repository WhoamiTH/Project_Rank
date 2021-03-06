import numpy as np
import math
import random
import sklearn.preprocessing as skpre
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
from time import clock
import copy
import operator

def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    temdata = np.hstack((tem[:, 2:7], tem[:, 9:-1]))
    temlabel = tem[:, -1]
    # data = temdata.astype(np.float).astype(np.int)
    data = temdata.astype(np.float)
    label = temlabel.astype(np.float).astype(np.int)
    return data, label


def group(Data):
    i = 1
    j = 1
    l = 1
    t = Data[0][1]
    Ds = {}
    Dl = {}
    Ds[1] = 0
    while j < len(Data):
        if (t != Data[j][1]):
            Dl[i] = l
            Ds[i + 1] = j
            i += 1
            l = 1
            t = Data[j][1]
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

def handleData_extend_mirror(Data, Label, Ds, Dl):
    handleDs = {}
    handleDl = {}
    temd = []
    teml = []
    handleDs[1] = 0
    for i in range(1, len(Ds) + 1):
        start = Ds[i]
        handleDl[i] = Dl[i] * (Dl[i]-1)
        if i < len(Ds):
            handleDs[i + 1] = handleDs[i] + handleDl[i]
        for j in range(Dl[i]):
            for t in range(Dl[i]):
                if j != t:
                    temd.append(data_extend(Data[start + j], Data[start + t]))
                    if Label[start + j] > Label[start + t]:
                        teml.append([-1])
                    else:
                        teml.append([1])
    arrtemd = np.array(temd)
    arrteml = np.array(teml)
    return arrtemd, arrteml, handleDs, handleDl

def handleData_extend_not_mirror(Data, Label, Ds, Dl):
    handleDs = {}
    handleDl = {}
    temd = []
    teml = []
    handleDs[1] = 0
    for i in range(1, len(Ds) + 1):
        start = Ds[i]
        handleDl[i] = int((Dl[i] * (Dl[i]-1))/2)
        if i < len(Ds):
            handleDs[i + 1] = handleDs[i] + handleDl[i]
        for j in range(Dl[i]):
            for t in range(j+1,Dl[i]):
                temd.append(data_extend(Data[start + j], Data[start + t]))
                if Label[start + j] > Label[start + t]:
                    teml.append([-1])
                else:
                    teml.append([1])
    arrtemd = np.array(temd)
    arrteml = np.array(teml)
    return arrtemd, arrteml, handleDs, handleDl

def data_diff(Data_1, Data_2):
    return list(Data_1 - Data_2)

def handleData_diff_mirror(Data, Label, Ds, Dl):
    handleDs = {}
    handleDl = {}
    temd = []
    teml = []
    handleDs[1] = 0
    for i in range(1, len(Ds) + 1):
        start = Ds[i]
        handleDl[i] = Dl[i] * (Dl[i] - 1)
        if i < len(Ds):
            handleDs[i + 1] = handleDs[i] + handleDl[i]
        for j in range(Dl[i]):
            for t in range(Dl[i]):
                if j != t:
                    temd.append(data_diff(Data[start + j], Data[start + t]))
                    if Label[start + j] > Label[start + t]:
                        teml.append([-1])
                    else:
                        teml.append([1])
    arrtemd = np.array(temd)
    arrteml = np.array(teml)
    return arrtemd, arrteml, handleDs, handleDl

def handleData_diff_not_mirror(Data, Label, Ds, Dl):
    handleDs = {}
    handleDl = {}
    temd = []
    teml = []
    handleDs[1] = 0
    for i in range(1, len(Ds) + 1):
        start = Ds[i]
        handleDl[i] = int((Dl[i] * (Dl[i] - 1))/2)
        if i < len(Ds):
            handleDs[i + 1] = handleDs[i] + handleDl[i]
        for j in range(Dl[i]):
            for t in range(j+1,Dl[i]):
                temd.append(data_diff(Data[start + j], Data[start + t]))
                if Label[start + j] > Label[start + t]:
                    teml.append([-1])
                else:
                    teml.append([1])
    arrtemd = np.array(temd)
    arrteml = np.array(teml)
    return arrtemd, arrteml, handleDs, handleDl

def exchange(test_y):
    ex_ty_list = []
    rank_ty = []
    for i in range(len(test_y)):
        ex_ty_list.append((int(test_y[i]),i+1))
    exed_ty = sorted(ex_ty_list)
    for i in exed_ty:
        rank_ty.append(i[1])
    return rank_ty


def calacc(rank, label):
    en = 0
    for i in range(len(rank)):
        if rank[i] == label[i]:
            en += 1
    return en / len(rank)


def count_general_pre(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if int(y_true[i]) == 1 and y_pred[i] > 0:
            tp += 1
        elif int(y_true[i]) == 1 and y_pred[i] < 0:
            fn += 1
        elif int(y_true[i]) == -1 and y_pred[i] > 0:
            fp += 1
    return tp, fp, fn


def count_top(y_true, y_pred):
    tp = 0
    exact = 0
    if top <= len(y_true):
        top_true = y_true[:top]
        top_pred = y_pred[:top]
    else:
        top_true = y_true
        top_pred = y_pred
    len_top = len(top_pred)
    # top_true = y_true[:top]
    # top_pred = y_pred[:top]
    # print(y_pred)
    # print(top_pred)
    for i in range(len_top):
        if top_pred[i] in top_true:
            tp += 1
            if top_pred[i] == top_true[i]:
                exact += 1
    group_pro = tp/top
    group_top_exact_accuracy = exact/top
    return group_pro, group_top_exact_accuracy


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def general_test(test_x,test_y,model,standardscaler):
    # standscaler = skpre.StandardScaler().fit(test_x)
    x = standardscaler.transform(test_x)
    general_predict = model.predict(x)
    # print(general_predict,end='\n\n')
    # print(test_y)
    # for i in range(len(test_y)):
    #     print(i,test_y[i])
    general_tp, general_fp, general_fn = count_general_pre(test_y, general_predict)
    general_precision, general_recall = precision_recall(general_tp, general_fp, general_fn)
    general_accuracy = (len(test_y) - general_fp - general_fn) / len(test_y)
    record.write(f"the general precision is {general_precision}\n")
    record.write(f"the general recall is {general_recall}\n")
    record.write(f"the general accuracy is {general_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n")
    # return standscaler

# def rank_the_group(group_x, length, reference, model, standscaler):
#     low = []
#     high = []
#     global handle_type
#     if len(reference) <= 1:
#         return reference
#     else:
#         # pivot = reference.pop()
#         # print('the reference is ',reference)
#         # print('the length is ', length)
#         pivot = random.choice(reference)
#         # if len(group_x) == length:
#         #     record.write(f"the pivot is          {pivot:2d}\n")
#         record.write(f"the pivot is          {pivot:2d}\n")
#         reference.remove(pivot)
#         # print("pivot is ",pivot)
#         for i in reference:
#             if handle_type == 'extend':
#                 t = data_extend(group_x[pivot-1], group_x[i-1])
#             else:
#                 t = data_diff(group_x[pivot-1], group_x[i-1])
#             t = np.array(t).reshape((1,-1))
#             standscaler.transform(t)
#             if model.predict(t) > 0:
#                 low.append(i)
#             else:
#                 high.append(i)
#     record.write('low  is ')
#     for i in low:
#         record.write(f'{i:2d}\t')
#     record.write('\n')
#     record.write('high is ')
#     for i in high:
#         record.write(f'{i:2d}\t')
#     record.write('\n')
#     low = rank_the_group(group_x, len(low), low, model,standscaler)
#     high = rank_the_group(group_x, len(high), high, model,standscaler)
#     high.append(pivot)
#     # print('high+low :',high+low)
#     return high + low

def compdata(group_x,model,standscaler,i):
    global handle_type
    if handle_type == 'extend':
        t = data_extend(group_x[i], group_x[i+1])
    else:
        t = data_diff(group_x[i], group_x[i+1])
    t = np.array(t).reshape((1,-1))
    # print(i)
    standscaler.transform(t)
    if model.predict(t) > 0:
        return True
    else:
        return False

def rank_the_group(group_x, length, reference, model, standscaler):
    if len(reference) <= 1:
        return reference
    else:
        for unsortednum in range(len(reference)-1,0,-1):
            record.write(f'the unsortednum is {unsortednum:2d}\n')
            for i in range(unsortednum):
                if compdata(group_x,model,standscaler,i):
                    temp = reference[i]
                    reference[i] = reference[i+1]
                    reference[i+1] = temp
            record.write(f"the {len(reference)-unsortednum:2d} order is       ")
            for item in reference:
                record.write(f'{item:2d}\t')
            record.write('\n')
    # print(reference)
    reference.reverse()
    # print(reference)
    return reference

def record_rank_reference(rank,reference):
    record.write("                      ")
    for m in range(1,len(rank)+1):
        record.write(f"{m:2d}\t")
    record.write("\n")
    record.write("the true rank is      ")
    for i in rank:
        record.write(f"{int(i):2d}\t")
    record.write("\n")
    record.write("the predict rank is   ")
    for t in reference:
        record.write(f"{int(t):2d}\t")
    record.write("\n")

def test_cycle(graph):
    cycle = []
    handled_set = []
    for i in range(len(graph)):
        dfs(graph,cycle,handled_set,i)

        handled_set.append(i)


def dfs(graph,cycle,handled_set,node):
    # print(cycle)
    if node not in handled_set:
        cycle.append(node)
        for edge in range(len(graph[node])):
            if graph[node][edge] == 1:
                if edge == cycle[0]:
                    record.write("there is a cycle:")
                    cycle.append(edge)
                    for item in cycle:
                        print(item,end='\t')
                        record.write(f'{(item + 1):2d}\t')
                    record.write('\n')
                    print()
                    cycle.pop()
                    continue
                if edge not in cycle:
                    dfs(graph,cycle,handled_set,edge)
        cycle.pop()

def group_test_relative(group_start,length,Data,Label,model,standscaler):
    global handle_type
    graph = []
    record.write(f'the start line is {group_start}\n')
    for j in range(length):
        edge_set = [-1 for i in range(length)]
        for t in range(length):
            if j != t:
                if handle_type == 'extend':
                    temd = data_extend(Data[group_start + j], Data[group_start + t])
                    temi = data_extend(Data[group_start + t], Data[group_start + j])
                else:
                    temd = data_diff(Data[group_start + j], Data[group_start + t])
                    temi = data_diff(Data[group_start + t], Data[group_start + j])
                temd = np.array(temd).reshape((1, -1))
                temi = np.array(temi).reshape((1, -1))
                standscaler.transform(temd)
                standscaler.transform(temi)
                print(model.predict_proba(temd))
                print(model.predict_proba(temi))
                if model.predict(temd) > 0:
                    edge_set[t] = 1
                    record.write(f'{j+1:2d}\t{t+1:2d}\t 1\t')
                else:
                    record.write(f'{j+1:2d}\t{t+1:2d}\t-1\t')
                if Label[group_start + j] > Label[group_start + t]:
                    record.write('-1\n')
                else:
                    record.write('1\n')
                if model.predict(temi) > 0:
                    record.write(f'{t+1:2d}\t{j+1:2d}\t 1\t')
                else:
                    record.write(f'{t+1:2d}\t{j+1:2d}\t-1\t')
                if Label[group_start + t] > Label[group_start + j]:
                    record.write('-1\n')
                    record.write('---------------------------------------\n')
                else:
                    record.write('1\n')
                    record.write('---------------------------------------\n')
                if model.predict(temd) == model.predict(temi):
                    record.write('there is a conflict!!!!!!!!!!!!!!!!!!!!\n')
        graph.append(edge_set)
    test_cycle(graph)


def group_test(Data, train_index_start,num_of_train, Ds, Dl, handleDs, handleDl, Label, model,standscaler):
    for group_index_start in range(1,len(handleDs)+1):
        if group_index_start<train_index_start:
            group_start = Ds[group_index_start]
        elif (group_index_start>=train_index_start and group_index_start<train_index_start+num_of_train):
            continue
        else:
            group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        group_end = group_start + length
        group_data = Data[group_start:group_end, :]
        group_label = Label[group_start:group_end]
        reference = [t for t in range(1, length + 1)]
        reference1 = rank_the_group(group_data, length, reference, model, standscaler)
        group_test_relative(group_start, length, Data, Label, model, standscaler)
        for time in range(100):
            reference = [t for t in range(1, length + 1)]
            random.shuffle(reference)
            record.write("the random order is   ")
            for t in reference:
                record.write(f"{int(t):2d}\t")
            record.write("\n")
            # print(reference)
            reference = rank_the_group(group_data, length, reference, model, standscaler)
            if not operator.eq(reference,reference1):
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                record.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                record.write("the predict rank is   ")
                for t in reference:
                    record.write(f"{int(t):2d}\t")
                record.write('\n')
                record.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

            reference1 = reference
            # print(reference)
            record.write("the predict rank is   ")
            for t in reference:
                record.write(f"{int(t):2d}\t")
            record.write("\n")
        # print(reference)
        group_rank = exchange(group_label)
        # print(group_rank)
        record_rank_reference(group_rank,reference)
        group_precision, group_top_exact_accuracy = count_top(group_rank, reference)
        group_accuracy = calacc(group_rank, reference)
        all_group_top_precision.append(group_precision)
        all_group_top_exact_accuracy.append(group_top_exact_accuracy)
        all_group_accuracy.append(group_accuracy)
        record.write(f"the group top precision is {group_precision}\n")
        record.write(f"the group top exact accuracy is {group_top_exact_accuracy} \n")
        record.write(f"the group accuracy is {group_accuracy}\n")
        record.write("-------------------------------------------------------------------------------------------\n")


def train_model(X, Y):
    start = clock()
    scaler = skpre.StandardScaler()
    scaler.fit(X)
    train_x = scaler.transform(X)
    global model_type
    if model_type == 'LR':
        model = sklin.LogisticRegression()
        model.fit(train_x,Y.flatten())
    if model_type == 'SVC':
        model = sksvm.SVC(C=0.1,kernel='linear')
        model.fit(train_x, Y.flatten())
    finish = clock()
    return model,finish-start,scaler

def generate_train_data(Ds,handleDs,handleDl,handledata,handlelabel,num_of_train):
    train_index_start = random.randint(1,len(Ds)-num_of_train+1)
    front = handleDs[train_index_start]
    # end = handleDs[train_index_start+num_of_train-1]
    end = handleDs[train_index_start+num_of_train-1]+handleDl[train_index_start+num_of_train-1]
    train_x = handledata[front:end,:]
    train_y = handlelabel[front:end,:]
    return train_index_start,train_x,train_y

# def generate_test_data(handledata,handlelabel,handleDs,handleDl,train_index_start,num_of_train):
#     front_end_index = handleDs[train_index_start]
#     end_start_index = handleDs[train_index_start+num_of_train-1]+handleDl[train_index_start+num_of_train-1]
#     test_x = np.vstack((handledata[0:front_end_index,:],handledata[end_start_index:,:]))
#     test_y = np.vstack((handlelabel[0:front_end_index,:],handlelabel[end_start_index:,:]))
#     return test_x,test_y

def generate_test_data(Ds,Dl,Data,Label,train_index_start,num_of_train):
    temd = []
    teml = []
    global handle_type
    for group_index_start in range(1,len(Ds)+1):
        if group_index_start<train_index_start:
            group_start = Ds[group_index_start]
        elif (group_index_start>=train_index_start and group_index_start<train_index_start+num_of_train):
            continue
        else:
            group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        for j in range(length):
            for t in range(length):
                if j != t:
                    if handle_type == 'extend':
                        temd.append(data_extend(Data[group_start + j], Data[group_start + t]))
                    else:
                        temd.append(data_diff(Data[group_start + j], Data[group_start + t]))
                    if Label[group_start + j] > Label[group_start + t]:
                        teml.append([-1])
                    else:
                        teml.append([1])
    test_x = np.array(temd)
    test_y = np.array(teml)
    return test_x,test_y

def calaverage():
    totle = len(all_group_top_precision)
    average_group_top_precision = sum(all_group_top_precision)/totle
    average_group_top_exact_accuracy = sum(all_group_top_exact_accuracy)/totle
    average_group_accuracy = sum(all_group_accuracy)/totle
    record.write(f"\nthe average group top precision is {average_group_top_precision}\n")
    record.write(f"the average group top exact accuracy is {average_group_top_exact_accuracy}\n")
    record.write(f"the average group accuracy is {average_group_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n\n")

def crossvalidation(percent, Ds, Dl, handleDs, handleDl, Data, Label, handledata, handlelabel):
    num_of_train = math.ceil(len(Ds) * percent)
    for i in range(rep_times):
        start = clock()
        train_index_start,train_x,train_y = generate_train_data(Ds,handleDs,handleDl,handledata,handlelabel,num_of_train)
        # print(train_index_start)
        test_x, test_y = generate_test_data(Ds, Dl, Data, Label, train_index_start, num_of_train)
        model,training_time,standardscaler = train_model(train_x, train_y)
        print(model)
        record.write(f"{str(model)}\n")
        record.write("-------------------------------------------------------------------------------------------\n")
        general_test(test_x,test_y,model,standardscaler)
        group_test(Data,train_index_start,num_of_train,Ds,Dl,handleDs,handleDl,Label,model,standardscaler)
        finish = clock()
        running_time = finish-start
        calaverage()
        global all_group_top_precision
        global all_group_top_exact_accuracy
        global all_group_accuracy
        all_group_top_precision = []
        all_group_top_exact_accuracy = []
        all_group_accuracy = []
        record.write(f"the {i+1} time training time is {training_time}\n")
        record.write(f"the {i+1} time running time is {running_time}\n")
        record.write("-------------------------------------------------------------------------------------------\n\n\n")

file_name = '/home/th/WorkSpace/python/python/Project/GData_test_800.csv'
# file_name = 'GData.csv'
# record_name = 'result_200.txt'
# path = '/home/th/WorkSpace/python/python/Project/Result/LR_diff_mirror_result_percent_'
# record = open(record_name,'w+')
model_type = 'LR'
# model_type = 'SVC'
# handle_type = "extend"
handle_type = "diff"
mirror_type = "mirror"
# mirror_type = "not_mirror"
path = '/home/th/WorkSpace/python/python/Project/Result/' + model_type + '_' + handle_type + '_' + mirror_type + '_result_percent_'
top = 3
# percent = 0.9
rep_times = 3
all_group_top_precision = []
all_group_top_exact_accuracy = []
all_group_accuracy = []
data, label = loadData(file_name)
dicstart, diclength = group(data)
the_start_line = open('/home/th/WorkSpace/python/python/Project/the_start_line.txt','w')
for i in dicstart:
    the_start_line.write(f'{dicstart[i]:5d}\t')
    for t in data[dicstart[i]]:
        the_start_line.write(str(t)+'\t')
    the_start_line.write('\n')



if mirror_type == 'mirror':
    if handle_type == 'extend':
        handledata, handlelabel, handledicstart, handlelength = handleData_extend_mirror(data, label, dicstart, diclength)
    if handle_type == 'diff':
        handledata, handlelabel, handledicstart, handlelength = handleData_diff_mirror(data, label, dicstart, diclength)
else:
    if handle_type == 'extend':
        handledata, handlelabel, handledicstart, handlelength = handleData_extend_not_mirror(data, label, dicstart, diclength)
    if handle_type == 'diff':
        handledata, handlelabel, handledicstart, handlelength = handleData_diff_not_mirror(data, label, dicstart, diclength)

# data_original, label_original = loadData(file_name)
# dicstart, diclength = group(data_original)
# if mirror_type == 'mirror':
#     if handle_type == 'extend':
#         handledata_original, handlelabel_original, handledicstart_original, handlelength_original = handleData_extend_mirror(data_original, label_original, dicstart, diclength)
#     if handle_type == 'diff':
#         handledata_original, handlelabel_original, handledicstart_original, handlelength_original = handleData_diff_mirror(data_original, label_original, dicstart, diclength)
# else:
#     if handle_type == 'extend':
#         handledata_original, handlelabel_original, handledicstart_original, handlelength_original = handleData_extend_not_mirror(data_original, label_original, dicstart, diclength)
#     if handle_type == 'diff':
#         handledata_original, handlelabel_original, handledicstart_original, handlelength_original = handleData_diff_not_mirror(data_original, label_original, dicstart, diclength)
# percent_list = []
# for i in range(1,10):
#     percent_list.append(i/10)
# for percent in percent_list:
for i in range(1,10):
    percent = i/10
    # data, label = loadData(file_name)
    # dicstart, diclength = group(data)
    # if mirror_type == 'mirror':
    #     if handle_type == 'extend':
    #         handledata, handlelabel, handledicstart, handlelength = handleData_extend_mirror(data, label, dicstart, diclength)
    #     if handle_type == 'diff':
    #         handledata, handlelabel, handledicstart, handlelength = handleData_diff_mirror(data, label, dicstart, diclength)
    # else:
    #     if handle_type == 'extend':
    #         handledata, handlelabel, handledicstart, handlelength = handleData_extend_not_mirror(data, label, dicstart, diclength)
    #     if handle_type == 'diff':
    #         handledata, handlelabel, handledicstart, handlelength = handleData_diff_not_mirror(data, label, dicstart, diclength)
    record_name = path+str(percent)+'.txt'
    record = open(record_name,'w')
    print(f"the percentage of training data is {percent}")
    # data = np.copy(data_original)
    # label = np.copy(label_original)
    # handledicstart = np.copy(handledicstart_original)
    # handlelength = np.copy(handlelength_original)
    # handledata = np.copy(handledata_original)
    # handlelabel = np.copy(handlelabel_original)
    crossvalidation(percent,dicstart, diclength, handledicstart, handlelength, data, label, handledata, handlelabel)
    print("\n\n\n")
    record.close()