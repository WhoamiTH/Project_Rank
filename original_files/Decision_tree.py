import numpy as np
import math
import random
import sklearn.preprocessing as skpre
import sklearn.svm as sksvm
import sklearn.tree as sktree
import sklearn.linear_model as sklin
from time import clock
import copy
import operator
from sklearn.decomposition import PCA

def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    tem_data = tem[:, 1:-1]
    tem_label = tem[:, -1]
    # data = tem_data.astype(np.float).astype(np.int)
    data = tem_data.astype(np.float)
    label = tem_label.astype(np.float).astype(np.int)
    return data, label


def group(Data):
    i = 1
    j = 1
    l = 1
    t = Data[0][2]
    Ds = {}
    Dl = {}
    Ds[1] = 0
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

def get_the_number(Data,column):
    item_Data = Data[:,column:column+1]
    item_Data = set([int(i) for i in item_Data])
    return len(item_Data)

def transform_the_column(Data,column,n_rows):
    length = get_the_number(Data,column)
    item_column = np.zeros((n_rows,length))
    the_column = Data[:,column:column+1]
    # print(the_column)
    for i in range(n_rows):
        item_column[i,int(the_column[i])-1] = 1
    return item_column

def transform_date_to_age(Data,n_rows):
    item_age = np.zeros((n_rows,1))
    for i in range(n_rows):
        item_age[i,0] = Data[i,1] - Data[i,23]
    return item_age

# def split_data(Data,divide_list,delete_list):
#     n_rows,n_column = Data.shape
#     item_age = transform_date_to_age(Data,n_rows)
#     tem_data = transform_the_column(Data,0,n_rows)
#     # print(tem_data.shape)
#     for column in range(1,n_column):
#         if column in divide_list:
#             item_column = transform_the_column(Data,column,n_rows)
#         elif column in delete_list:
#             continue
#         else:
#             item_column = Data[:,column:column+1]
#         tem_data = np.hstack((tem_data,item_column))
#         # print(tem_data.shape)
#     tem_data = np.hstack((tem_data,item_age))
#     # print(tem_data.shape)
#     return tem_data

def split_data(Data,divide_list,delete_list):
    n_rows,n_column = Data.shape
    item_age = transform_date_to_age(Data,n_rows)
    tem_data = np.delete(Data, delete_list, axis=1)
    tem_data = np.hstack((tem_data,item_age))
    # print(tem_data.shape)
    return tem_data

def data_extend(Data_1, Data_2):
    m = list(Data_1)
    n = list(Data_2)
    return m + n

def handleData_extend_mirror(Data, Label, start, length):
    temd = []
    teml = []
    for j in range(length):
        for t in range(length):
            if j != t:
                temd.append(data_extend(Data[start + j], Data[start + t]))
                if Label[start + j] > Label[start + t]:
                    teml.append([-1])
                else:
                    teml.append([1])
    return temd, teml

def handleData_extend_not_mirror(Data, Label, start, length):
    temd = []
    teml = []
    for j in range(length):
        for t in range(j+1,length):
            temd.append(data_extend(Data[start + j], Data[start + t]))
            if Label[start + j] > Label[start + t]:
                teml.append([-1])
            else:
                teml.append([1])
    return temd, teml

def condense_data(Data):
    pca = PCA(n_components=18)
    pca.fit(Data)
    new_data = pca.transform(Data)
    return new_data,pca

def standardize_data(Data):
    scaler = skpre.StandardScaler()
    scaler.fit(Data)
    return scaler

# def PCA_stand_data(Data):
#     new_data,pca = condense_data(Data)
#     scaler = standardize_data(new_data)
#     return pca,scaler

def PCA_stand_data(Data):
    scaler = standardize_data(Data)
    return scaler

# def transform_data(Data,pca,scaler):
#     new_data = pca.transform(Data)
#     new_data = scaler.transform(new_data)
#     return new_data

def transform_data(Data,scaler):
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


def general_test(test_data,test_label,model):
    general_predict = model.predict(test_data)
    general_tp, general_fp, general_fn = count_general_pre(test_label, general_predict)
    general_precision, general_recall = precision_recall(general_tp, general_fp, general_fn)
    general_accuracy = (len(test_label) - general_fp - general_fn) / len(test_label)
    record.write(f"the general precision is {general_precision}\n")
    record.write(f"the general recall is {general_recall}\n")
    record.write(f"the general accuracy is {general_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n")

def rank_the_group(group_data, length, reference, model):
    low = []
    high = []
    global handle_type
    if len(reference) <= 1:
        return reference
    else:
        # pivot = reference.pop()
        # print('the reference is ',reference)
        # print('the length is ', length)
        pivot = random.choice(reference)
        # if len(group_data) == length:
        #     record.write(f"the pivot is          {pivot:2d}\n")
        record.write(f"the pivot is          {pivot:2d}\n")
        reference.remove(pivot)
        # print("pivot is ",pivot)
        for i in reference:
            t = data_extend(group_data[pivot-1], group_data[i-1])
            t = np.array(t).reshape((1,-1))
            if model.predict(t) > 0:
                low.append(i)
            else:
                high.append(i)
    record.write('low  is ')
    for i in low:
        record.write(f'{i:2d}\t')
    record.write('\n')
    record.write('high is ')
    for i in high:
        record.write(f'{i:2d}\t')
    record.write('\n')
    low = rank_the_group(group_data, len(low), low, model)
    high = rank_the_group(group_data, len(high), high, model)
    high.append(pivot)
    # print('high+low :',high+low)
    return high + low

# def compdata(group_data,model,standscaler,i):
#     global handle_type
#     if handle_type == 'extend':
#         t = data_extend(group_data[i], group_data[i+1])
#     else:
#         t = data_diff(group_data[i], group_data[i+1])
#     t = np.array(t).reshape((1,-1))
#     # print(i)
#     standscaler.transform(t)
#     if model.predict(t) > 0:
#         return True
#     else:
#         return False
#
# def rank_the_group(group_data, length, reference, model, standscaler):
#     if len(reference) <= 1:
#         return reference
#     else:
#         for unsortednum in range(len(reference)-1,0,-1):
#             record.write(f'the unsortednum is {unsortednum:2d}\n')
#             for i in range(unsortednum):
#                 if compdata(group_data,model,standscaler,i):
#                     temp = reference[i]
#                     reference[i] = reference[i+1]
#                     reference[i+1] = temp
#             record.write(f"the {len(reference)-unsortednum:2d} order is       ")
#             for item in reference:
#                 record.write(f'{item:2d}\t')
#             record.write('\n')
#     # print(reference)
#     reference.reverse()
#     # print(reference)
#     return reference

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
    if node not in handled_set:
        cycle.append(node)
        for edge in range(len(graph[node])):
            if graph[node][edge] == 1:
                if edge == cycle[0]:
                    record.write("there is a cycle:")
                    cycle.append(edge)
                    for item in cycle:
                        # print(item,end='\t')
                        record.write(f'{(item + 1):2d}\t')
                    record.write('\n')
                    # print()
                    cycle.pop()
                    continue
                if edge not in cycle:
                    dfs(graph,cycle,handled_set,edge)
        cycle.pop()

def group_test_relative(group_start,length,Data,Label,model):
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
                # else:
                #     temd = data_diff(Data[group_start + j], Data[group_start + t])
                #     temi = data_diff(Data[group_start + t], Data[group_start + j])
                temd = np.array(temd).reshape((1, -1))
                temi = np.array(temi).reshape((1, -1))
                # print(model.predict_proba(temd))
                # print(model.predict_proba(temi))
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


def group_test(Data, Label, Ds, Dl, train_index_start, num_of_train, model):
    for group_index_start in range(1,len(Ds)+1):
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
        # reference = [t for t in range(1, length + 1)]
        # reference1 = rank_the_group(group_data, length, reference, model)
        # group_test_relative(group_start, length, Data, Label, model)
        for time in range(100):
            reference = [t for t in range(1, length + 1)]
            random.shuffle(reference)
            record.write("the random order is   ")
            for t in reference:
                record.write(f"{int(t):2d}\t")
            record.write("\n")
            # print(reference)
            reference = rank_the_group(group_data, length, reference, model)
            # if not operator.eq(reference,reference1):
            #     # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #     record.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            #     record.write("the predict rank is   ")
            #     for t in reference:
            #         record.write(f"{int(t):2d}\t")
            #     record.write('\n')
            #     record.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

            # reference1 = reference
            # print(reference)
            # record.write("the predict rank is   ")
            # for t in reference:
            #     record.write(f"{int(t):2d}\t")
            # record.write("\n")
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
    global model_type
    if model_type == 'LR':
        model = sklin.LogisticRegression()
        model.fit(X,Y.flatten())
    if model_type == 'SVC':
        # model = sksvm.SVC(C=0.1,kernel='rbf')
        model = sksvm.SVC(C=0.1,kernel='poly')
        model.fit(X, Y.flatten())
    if model_type == 'DT':
        model = sktree.DecisionTreeClassifier()
        model.fit(X,Y.flatten())
    finish = clock()
    return model,finish-start

def generate_primal_train_data(Data,Label,Ds,Dl,num_of_train):
    train_index_start = random.randint(1,len(Ds)-num_of_train+1)
    front = Ds[train_index_start]
    end = Ds[train_index_start+num_of_train-1]+Dl[train_index_start+num_of_train-1]
    train_x = Data[front:end,:]
    train_y = Label[front:end]
    return train_index_start,train_x,train_y

def generate_all_data(Ds,Dl,Data,Label,train_index_start,num_of_train):
    tem_data_train = []
    tem_label_train = []
    tem_data_test = []
    tem_label_test = []
    global mirror_type
    for group_index_start in range(1,len(Ds)+1):
        group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        if group_index_start<train_index_start:
            temd,teml = handleData_extend_mirror(Data,Label,group_start,length)
            tem_data_test = tem_data_test + temd
            tem_label_test = tem_label_test + teml
        elif (group_index_start>=train_index_start and group_index_start<train_index_start+num_of_train):
            if mirror_type == 'mirror':
                temd, teml = handleData_extend_mirror(Data, Label, group_start, length)
            else:
                temd, teml = handleData_extend_not_mirror(Data, Label, group_start, length)
            tem_data_train = tem_data_train + temd
            tem_label_train = tem_label_train + teml
        else:
            temd, teml = handleData_extend_mirror(Data, Label, group_start, length)
            tem_data_test = tem_data_test + temd
            tem_label_test = tem_label_test + teml
    train_data = np.array(tem_data_train)
    train_label = np.array(tem_label_train)
    test_data = np.array(tem_data_test)
    test_label = np.array(tem_label_test)
    return train_data, train_label, test_data, test_label

def calaverage():
    totle = len(all_group_top_precision)
    average_group_top_precision = sum(all_group_top_precision)/totle
    average_group_top_exact_accuracy = sum(all_group_top_exact_accuracy)/totle
    average_group_accuracy = sum(all_group_accuracy)/totle
    record.write(f"\nthe average group top precision is {average_group_top_precision}\n")
    record.write(f"the average group top exact accuracy is {average_group_top_exact_accuracy}\n")
    record.write(f"the average group accuracy is {average_group_accuracy}\n")
    record.write("-------------------------------------------------------------------------------------------\n\n")

def crossvalidation(percent, Ds, Dl, Data, Label):
    num_of_train = math.ceil(len(Ds) * percent)
    for i in range(rep_times):
        start = clock()
        train_index_start,train_x,train_y = generate_primal_train_data(Data,Label,Ds,Dl,num_of_train)
        # pca,scaler = PCA_stand_data(train_x)
        scaler = PCA_stand_data(train_x)
        # new_data = transform_data(Data,pca,scaler)
        new_data = transform_data(Data, scaler)
        train_data,train_label,test_data,test_label= generate_all_data(Ds, Dl, new_data, Label, train_index_start, num_of_train)
        model,training_time = train_model(train_data, train_label)
        print(model)
        record.write(f"{str(model)}\n")
        record.write("-------------------------------------------------------------------------------------------\n")
        general_test(test_data,test_label,model)
        group_test(new_data,Label,Ds,Dl,train_index_start,num_of_train,model)
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
delete_list = [1,2,7,15,17,19,23]
divide_list = [0,8,9,12,13,20,22]
# model_type = 'LR'
# model_type = 'SVC'
model_type = 'DT'
handle_type = "extend"
# handle_type = "diff"
# mirror_type = "mirror"
mirror_type = "not_mirror"
path = '/home/th/WorkSpace/python/python/Project/Result/' + model_type + '_' + handle_type + '_' + mirror_type + '_result_percent_'
top = 2
rep_times = 3
all_group_top_precision = []
all_group_top_exact_accuracy = []
all_group_accuracy = []
data, label = loadData(file_name)
dicstart, diclength = group(data)
data = split_data(data,divide_list,delete_list)
for i in range(1,10):
    percent = i/10
    record_name = path+str(percent)+'.txt'
    record = open(record_name,'w')
    print(f"the percentage of training data is {percent}")
    crossvalidation(percent,dicstart, diclength, data, label)
    print("\n\n\n")
    record.close()