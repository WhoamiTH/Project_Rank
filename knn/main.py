import math
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.tree as sktree
from time import clock
import handle_data
import predict_test
import sys
import numpy as np
import random


def train_model(train_data, train_label, test_data, test_label):
    start = clock()
    global model_type
    if model_type == 'LR':
        model = sklin.LogisticRegression()
        model.fit(train_data,train_label.flatten())
    if model_type == 'SVC':
        model = sksvm.SVC(C=0.1,kernel='rbf')
        # model = sksvm.SVC(C=0.1,kernel='poly')
        model.fit(train_data, train_label.flatten())
    if model_type == 'DT':
        model = sktree.DecisionTreeClassifier()
        model.fit(train_data, train_label.flatten())
    finish = clock()
    return model, finish-start

def crossvalidation(percent, target_rate, Ds, Dl, Data, Label, record, knn_list, knn_label_list, final_knn_list, distance_matric):
    global pca_or_not
    global kernelpca_or_not
    global num_of_components
    global positive_value
    global negative_value
    global threshold_value
    global mirror_type
    num_of_train = math.ceil(len(Ds) * percent)
    end_index_train = Ds[num_of_train-1] + Dl[num_of_train-1]
    primal_label = Label[0:end_index_train]
    for i in range(rep_times):
        all_group_top_precision = []
        all_group_recall = []
        all_group_top_exact_accuracy = []

        start = clock()
        train_data, train_label = handle_data.generate_train_data(Data, Label, primal_label, target_rate, knn_list, knn_label_list, final_knn_list, distance_matric)
        new_data, train_x = handle_data.standarize_PCA_data(train_data, Data, pca_or_not, kernelpca_or_not, num_of_components)
        test_data,test_label= handle_data.generate_test_data(new_data, Label, Ds, num_of_train)
        model,training_time = train_model(train_data, train_label, test_data, test_label)
        finish = clock()
        print(model)
        record.write(str(model) + "\n")
        record.write("-------------------------------------------------------------------------------------------\n")
        predict_test.general_test(test_data, test_label, model, positive_value, negative_value, threshold_value, record)
        all_group_top_precision, all_group_top_exact_accuracy, all_group_exact_accuracy = \
            predict_test.group_test(new_data, Label, Ds, Dl, num_of_train, model,
                                    top, all_group_top_precision,all_group_recall, all_group_top_exact_accuracy, record)
        running_time = finish-start
        predict_test.cal_average(all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, record)
        record.write("the {0} time training time is {1}\n".format(i+1,training_time))
        record.write("the {0} time running time is {1}\n".format(i+1,running_time))
        record.write("-------------------------------------------------------------------------------------------\n\n\n")

def process(record_name, percent, target_rate, Ds, Dl, Data, Label, knn_list, knn_label_list, final_knn_list, distance_matric):
    record = open(record_name, 'w')
    print("the percentage of training data is {0}".format(percent))
    crossvalidation(percent, target_rate, Ds, Dl, Data, Label, record, knn_list, knn_label_list, final_knn_list, distance_matric)
    record.close()
    handle_data.append_file(record_name)
    print("\n\n\n")


def set_para():
    global model_type
    # global mirror_type
    # global top
    # global rep_times
    # global positive_value
    # global negative_value
    # global threshold_value
    global kernelpca_or_not
    global pca_or_not
    global num_of_components
    global percentage

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'model_type':
            model_type = para[1].upper()
        # if para[0] == 'mirror_type':
        #     mirror_type = para[1]
        # if para[0] == 'top':
        #     top = int(para[1])
        # if para[0] == 'rep_times':
        #     rep_times = int(para[1])
        # if para[0] == 'positive_value':
        #     positive_value = int(para[1])
        # if para[0] == 'negative_value':
        #     negative_value = int(para[1])
        # if para[0] == 'threshold_value':
        #     threshold_value = int(para[1])
        if para[0] == 'kernelpca':
            if para[1] == 'True':
                kernelpca_or_not = True
            else:
                kernelpca_or_not = False
        if para[0] == 'pca':
            if para[1] == 'True':
                pca_or_not = True
            else:
                pca_or_not = False
        if para[0] == 'num_of_components':
            num_of_components = int(para[1])
        if para[0] == 'percentage':
            percentage = int(para[1])
    if kernelpca_or_not and pca_or_not:
        pca_or_not = True
        kernelpca_or_not = False






# -------------------------------------global parameters---------------------------------------------------------------
model_type = 'LR'
# model_type = 'SVC'
# model_type = 'DT'
# mirror_type = "mirror"
mirror_type = "not_mirror"
top = 3
rep_times = 3
positive_value = 1
negative_value = 0
threshold_value = 0.5
kernelpca_or_not = False
pca_or_not = False
num_of_components = 20
percentage = 0
target_rate = 1.5
k = 7

# ----------------------------------set parameters--------------------------------------------------------------------
set_para()
file_name = 'GData_new.csv'
path = model_type

# ----------------------------------start processing--------------------------------------------------------------------
data, label = handle_data.loadData(file_name)
dicstart, diclength = handle_data.group(data)

record_name = path


if percentage != 0:
    record_name += '_result_percent_' + str(percentage)
    if pca_or_not:
        record_name += '_pca_' + str(num_of_components)
    if kernelpca_or_not:
        record_name += '_kernel_pca_' + str(num_of_components)
    record_name += '.txt'
    process(record_name, percentage, dicstart, diclength, data, label)
else:
    for i in range(1,10):
        percentage = i/10
        # percent = 0.9
        # record_name = path + str(percent) + '_' + str(i) + '.txt'
        distance_matric_name = 'all_distance.npz'
        distance_matric = handle_data.load_npz_data(distance_matric_name)
        knn_list_name = str(percentage) + '_' + str(k) + '_knn_list.npz'
        knn_list = handle_data.load_npz_data(knn_list_name)
        knn_label_list_name = str(percentage) + '_' + str(k) +  '_knn_label_list.npz'
        knn_label_list = handle_data.load_npz_data(knn_label_list_name)
        final_knn_list_name = str(percentage) + '_' + str(k) +  '_final_knn_list.npz'
        final_knn_list = handle_data.load_npz_data(final_knn_list_name)
        final_knn_list = np.ravel(final_knn_list.sum(axis=0))
        final_knn_list = final_knn_list.tolist()


        record_name += '_result_percent_' + str(percentage)
        if pca_or_not:
            record_name += '_pca_' + str(num_of_components)
        if kernelpca_or_not:
            record_name += '_kernel_pca_' + str(num_of_components)
        record_name += '.txt'
        process(record_name, percentage, target_rate, dicstart, diclength, data, label, knn_list, knn_label_list, final_knn_list, distance_matric)
        record_name = path
