import math
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.tree as sktree
from time import clock
import handle_data
import predict_test
import sys


def train_model(train_data, train_label, test_data, test_label):
    start = clock()
    global model_type
    if model_type == 'LR':
        model = sklin.LogisticRegression()
        model.fit(train_data,train_label.flatten())
    if model_type == 'SVC':
        model = sksvm.SVC(C=0.1,kernel='linear')
        # model = sksvm.SVC(C=0.1,kernel='rbf')
        # model = sksvm.SVC(C=0.1,kernel='poly')
        model.fit(train_data, train_label.flatten())
    if model_type == 'DT':
        model = sktree.DecisionTreeClassifier()
        model.fit(train_data, train_label.flatten())
    finish = clock()
    return model, finish-start

def crossvalidation(percent, Ds, Dl, Data, Label, record):
    global pca_or_not
    global kernelpca_or_not
    global num_of_components
    global positive_value
    global negative_value
    global threshold_value
    global mirror_type
    #-------------------------------------------------------
    global result_number
    global winner_number
    #-------------------------------------------------------
    num_of_train = math.ceil(len(Ds) * percent)
    for i in range(rep_times):
        all_group_top_precision = []
        all_group_recall = []
        all_group_top_exact_accuracy = []
        all_group_exact_accuracy = []
        start = clock()
        train_x,train_y = handle_data.generate_primal_train_data(Data,Label,Ds,Dl,num_of_train)
        new_data = handle_data.standarize_PCA_data(train_x, Data, pca_or_not, kernelpca_or_not, num_of_components)
        train_data,train_label,test_data,test_label= handle_data.generate_all_data(Ds, Dl, new_data, Label, num_of_train, mirror_type, positive_value, negative_value)
        model,training_time = train_model(train_data, train_label, test_data, test_label)
        finish = clock()
        print(model)


        record.write(str(model) + "\n")
        record.write("-------------------------------------------------------------------------------------------\n")
        # predict_test.general_test(test_data, test_label, model, positive_value, negative_value, threshold_value, record)
        all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy = \
            predict_test.group_test(new_data, Label, Ds, Dl, num_of_train, model, threshold_value, result_number, winner_number,
                                    all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, record)
        running_time = finish-start
        predict_test.cal_average(all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, record)
        record.write("the {0} time training time is {1}\n".format(i+1,training_time))
        record.write("the {0} time running time is {1}\n".format(i+1,running_time))
        record.write("-------------------------------------------------------------------------------------------\n\n\n")

def process(record_name, percent, Ds, Dl, Data, Label):
    record = open(record_name, 'w')
    print("the percentage of training data is {0}".format(percent))
    crossvalidation(percent, Ds, Dl, Data, Label, record)
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
mirror_type = "mirror"
# mirror_type = "not_mirror"
top = 3
rep_times = 3
positive_value = 1
negative_value = -1
threshold_value = 0
kernelpca_or_not = False
pca_or_not = False
num_of_components = 20
percentage = 0

#------------------------
result_number = 8
winner_number = 3
#------------------------

# ----------------------------------set parameters--------------------------------------------------------------------
set_para()
file_name = 'GData_new.csv'
path = model_type

# ----------------------------------start processing--------------------------------------------------------------------
data, label = handle_data.loadData(file_name)
dicstart, diclength = handle_data.group(data)

record_name = path


if percentage != 0:
    record_name += mirror_type + '_result_percent_' + str(percentage)
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
        record_name += '_' + mirror_type + '_result_percent_' + str(percentage)
        if pca_or_not:
            record_name += '_pca_' + str(num_of_components)
        if kernelpca_or_not:
            record_name += '_kernel_pca_' + str(num_of_components)
        record_name += '.txt'
        process(record_name, percentage, dicstart, diclength, data, label)
        record_name = path
