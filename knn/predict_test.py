# This code is prepared for knn method

import numpy as np
import random
import handle_data


def count_general_pre(y_true, y_pred, positive_value, negative_value, threshold_value):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if int(y_true[i]) == positive_value and y_pred[i] > threshold_value:
            tp += 1
        elif int(y_true[i]) == positive_value and y_pred[i] < threshold_value:
            fn += 1
        elif int(y_true[i]) == negative_value and y_pred[i] > threshold_value:
            fp += 1
    return tp, fp, fn


def count_top(y_true, y_pred, top):
    tp = 0
    exact = 0
    if top <= len(y_true):
        top_true = y_true[:top]
        len_exact = top
    else:
        top_true = y_true
        len_exact = len(y_true)
    length = len(top_true)
    len_pre = len(y_pred)
    for i in range(length):
        if top_true[i] in y_pred:
            tp += 1
            if i < len_pre:
                if top_true[i] == y_pred[i]:
                    exact += 1
    if len_pre == 0:
        group_top_pre = 0
        group_recall = 0
        group_top_exact_accuracy = 0
    else:
        group_top_pre = tp/len_pre
        group_recall = tp/len_exact
        group_top_exact_accuracy = exact/len_exact
    return group_top_pre, group_recall, group_top_exact_accuracy


def precision_recall(tp, fp, fn):
    if (tp+fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp+fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    return precision, recall


def general_test(test_data, test_label, model, positive_value, negative_value, threshold_value, record):
    general_predict = model.predict(test_data)
    general_tp, general_fp, general_fn = count_general_pre(test_label, general_predict, positive_value, negative_value, threshold_value)
    general_precision, general_recall = precision_recall(general_tp, general_fp, general_fn)
    general_accuracy = (len(test_label) - general_fp - general_fn) / len(test_label)
    # --------------------------------------------------------------------
    record.write("the general true positive is {0}\n".format(general_tp))
    record.write("the general false positive is {0}\n".format(general_fp))
    record.write("the general false negative is {0}\n".format(general_fn))
    general_tn = len(test_label) - general_fp - general_fn - general_tp
    record.write("the general true negative is {0}\n".format(general_tn))
    # --------------------------------------------------------------------
    record.write("the general precision is {0}\n".format(general_precision))
    record.write("the general recall is {0}\n".format(general_recall))
    record.write("the general accuracy is {0}\n".format(general_accuracy))
    record.write("-------------------------------------------------------------------------------------------\n")


def record_middle_result(name, list, record):
    record.write(name)
    for i in list:
        record.write("{0:2d}\t".format(i))
    record.write('\n')




def record_rank_reference(reference, rank, predict_rank, record):
    t = [i for i in range(1,len(rank)+1)]
    record_middle_result('                      ', t, record)
    record_middle_result('the random order is   ', reference, record)
    record_middle_result('the true rank is      ', rank, record)
    record_middle_result('the predict rank is   ', predict_rank, record)


def group_test(Data, Label, Ds, Dl, num_of_train, model, top, all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, record):
    for group_index_start in range(num_of_train, len(Ds)):
        group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        group_end = group_start + length
        group_data = Data[group_start:group_end, :]
        group_label = Label[group_start:group_end]
        predict_result = model.predict(group_data)
        result = np.where(predict_result==1)
        result = result[0].tolist()
        group_rank = handle_data.exchange(group_label)
        group_top_precision, group_recall, group_top_exact_accuracy = count_top(group_rank, result, top)

        all_group_top_precision.append(group_top_precision)
        all_group_recall.append(group_recall)
        all_group_top_exact_accuracy.append(group_top_exact_accuracy)

        record.write("the group top precision is {0}\n".format(group_top_precision))
        record.write("the group recall is {0}\n".format(group_recall))
        record.write("the group top exact accuracy is {0}\n".format(group_top_exact_accuracy))
        record.write("-------------------------------------------------------------------------------------------\n")
    return all_group_top_precision, all_group_recall, all_group_top_exact_accuracy


def cal_average(all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, record):
    totle = len(all_group_top_precision)
    average_group_top_precision = sum(all_group_top_precision)/totle
    average_group_recall = sum(all_group_recall)/totle
    average_group_top_exact_accuracy = sum(all_group_top_exact_accuracy)/totle

    # record.write("\nthe average group top precision is {0}\n".format(average_group_top_precision))
    record.write("the average group top precision is {0}\n".format(average_group_top_precision))
    record.write("the average group recall is {0}\n".format(average_group_recall))
    record.write("the average group top exact accuracy is {0}\n".format(average_group_top_exact_accuracy))

    # record.write("-------------------------------------------------------------------------------------------\n\n")

