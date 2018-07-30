# This code for relative_relationship_version_2

import numpy as np
import random
import handle_data


def calacc(rank, label):
    en = 0
    for i in range(len(rank)):
        if rank[i] == label[i]:
            en += 1
    return en / len(rank)


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


# def count_top(y_true, y_pred, top):
#     tp = 0
#     exact = 0
#     if top <= len(y_true):
#         top_true = y_true[:top]
#         top_pred = y_pred[:top]
#     else:
#         top_true = y_true
#         top_pred = y_pred
#     len_top = len(top_pred)
#     for i in range(len_top):
#         if top_pred[i] in top_true:
#             tp += 1
#             if top_pred[i] == top_true[i]:
#                 exact += 1
#     group_top_pre = tp/top
#     group_top_exact_accuracy = exact/top
#     return group_top_pre, group_recall, group_top_exact_accuracy


def count_top(y_true, y_pred, result_number, winner_number):
    tp = 0
    exact = 0
    if result_number <= len(y_true):
        top_true = y_true[:winner_number]
        top_pred = y_pred[:result_number]
    elif len(y_true) >= winner_number:
        top_true = y_true[:winner_number]
        top_pred = y_pred
    else:
        top_true = y_true
        top_pred = y_pred
    len_top = len(top_pred)
    for i in range(len_top):
        if top_pred[i] in top_true:
            tp += 1
            if i < winner_number:
                if top_pred[i] == top_true[i]:
                    exact += 1
    if result_number == len_top:
        group_pre = tp/result_number
        group_recall = tp/winner_number
        group_top_exact_accuracy = exact/winner_number
    elif len_top >= winner_number:
        group_pre = tp/len_top
        group_recall = tp/winner_number
        group_top_exact_accuracy = exact/winner_number
    else:
        group_pre = tp/len_top
        group_recall = tp/len_top
        group_top_exact_accuracy = exact/len_top
    return group_pre, group_recall, group_top_exact_accuracy




def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp)
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


# def rank_the_group(group_data, reference, model, threshold, record):
#     low = []
#     high = []
#     if len(reference) <= 1:
#         return reference
#     else:
#         pivot = random.choice(reference)
#         # record.write(f"the pivot is          {pivot:2d}\n")
#         reference.remove(pivot)
#         for i in reference:
#             t = handle_data.data_extend(group_data[pivot-1], group_data[i-1])
#             t = np.array(t).reshape((1,-1))
#             if model.predict(t) > threshold:
#                 low.append(i)
#             else:
#                 high.append(i)
#     # record_middle_result('low', low, record)
#     # record_middle_result('high', high, record)
#     low = rank_the_group(group_data, low, model, threshold, record)
#     high = rank_the_group(group_data, high, model, threshold, record)
#     high.append(pivot)
#     return high + low



def rank_the_group(group_data, reference, model, threshold, result_number, record):
    if len(reference) <= result_number:
        return reference
    else:
        tem = [reference.pop()]
        for each in reference:
            for item in range(len(tem)):
                t = handle_data.data_extend(group_data[each-1], group_data[tem[item]-1])
                t = np.array(t).reshape((1,-1))
                if model.predict(t) > threshold:
                    tem.insert(item, each)
                    break
                else:
                    if item == len(tem)-1:
                        tem.append(each)
                        break
        return tem



def record_rank_reference(reference, rank, predict_rank, record):
    t = [i for i in range(1,len(rank)+1)]
    record_middle_result('                      ', t, record)
    record_middle_result('the random order is   ', reference, record)
    record_middle_result('the true rank is      ', rank, record)
    record_middle_result('the predict rank is   ', predict_rank, record)


def group_test(Data, Label, Ds, Dl, num_of_train, model, threshold_value, result_number, winner_number, all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy, record):
    for group_index_start in range(num_of_train,len(Ds)):
        group_start = Ds[group_index_start]
        length = Dl[group_index_start]
        group_end = group_start + length
        group_data = Data[group_start:group_end, :]
        group_label = Label[group_start:group_end]
        reference = [t for t in range(1, length + 1)]
        random.shuffle(reference)
        predict_rank = rank_the_group(group_data, reference, model, threshold_value, result_number, record)
        group_rank = handle_data.exchange(group_label)
        # reference = [t for t in range(1, length + 1)]
        # record_rank_reference(reference, group_rank, predict_rank, record)
        # --------------------------------------------------------------------
        # group_test_relative(group_data, group_label, length, model, threshold_value, record)
        # --------------------------------------------------------------------
        group_top_precision, group_recall, group_top_exact_accuracy = count_top(group_rank, predict_rank, result_number, winner_number)
        group_exact_accuracy = calacc(group_rank, predict_rank)
        all_group_top_precision.append(group_top_precision)
        all_group_recall.append(group_recall)
        all_group_top_exact_accuracy.append(group_top_exact_accuracy)
        all_group_exact_accuracy.append(group_exact_accuracy)

        # record.write("the group top precision is {0}\n".format(group_top_precision))
        # record.write("the group recall is {0}\n".format(group_recall))
        # record.write("the group top exact accuracy is {0}\n".format(group_top_exact_accuracy))
        # record.write("the group accuracy is {0}\n".format(group_exact_accuracy))
        # record.write("-------------------------------------------------------------------------------------------\n")
    return all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_exact_accuracy





def cal_average(all_group_top_precision, all_group_recall, all_group_top_exact_accuracy, all_group_accuracy, record):
    totle = len(all_group_top_precision)
    average_group_top_precision = sum(all_group_top_precision)/totle
    average_group_recall = sum(all_group_recall)/totle
    average_group_top_exact_accuracy = sum(all_group_top_exact_accuracy)/totle
    average_group_accuracy = sum(all_group_accuracy)/totle
    # record.write("\nthe average group top precision is {0}\n".format(average_group_top_precision))
    record.write("the average group top precision is {0}\n".format(average_group_top_precision))
    record.write("the average group recall is {0}\n".format(average_group_recall))
    record.write("the average group top exact accuracy is {0}\n".format(average_group_top_exact_accuracy))
    record.write("the average group accuracy is {0}\n".format(average_group_accuracy))
    # record.write("-------------------------------------------------------------------------------------------\n\n")