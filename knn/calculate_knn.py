import handle_data_for_distance
from scipy import sparse
import math
import numpy as np
import sys

# ----------------------------------start processing--------------------------------------------------------------------
def save_by_npz(target, file_name):
    tem = np.array(target)
    stem = sparse.csr_matrix(tem)
    sparse.save_npz(file_name, stem)



def calculate_knn(percentage, k, selected_threshold, noisy_threshold):
    file_name = 'GData_new.csv'
    # file_name = 'GData_test_10.csv'
    distance_file_name = 'all_distance.npz'

    data, label = handle_data_for_distance.loadData(file_name)
    dicstart, diclength = handle_data_for_distance.group(data)

    distance_matric = sparse.load_npz(distance_file_name)
    distance_matric = distance_matric.toarray()

    num_of_train = math.ceil(len(dicstart) * percentage)

    end_index_train = dicstart[num_of_train-1] + diclength[num_of_train-1]

    training_distance_matric = distance_matric[0:end_index_train, 0:end_index_train]
    train_label = label[0:end_index_train]


    knn_list = handle_data_for_distance.create_knn_list(training_distance_matric, k)
    knn_list_name = str(percentage) + '_knn_list.npz'
    save_by_npz(knn_list, knn_list_name)

    knn_label_list = handle_data_for_distance.transform_knn_list_to_label(knn_list, label)
    knn_label_list_name = str(percentage) + '_knn_label_list.npz'
    save_by_npz(knn_label_list, knn_label_list_name)

    final_knn_list = handle_data_for_distance.eliminate_noisy(knn_label_list, selected_threshold, noisy_threshold, train_label)
    final_knn_list_name = str(percentage) + '_final_knn_list.npz'
    save_by_npz(final_knn_list,final_knn_list_name)


if __name__ == '__main__' :
    k = 6
    selected_threshold = 0
    noisy_threshold = 6
    for i in range(1,10):
        percentage = i/10
        calculate_knn(percentage, k, selected_threshold, noisy_threshold)
