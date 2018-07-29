import numpy as np
import csv


def loadData(file_name):
    tem = np.loadtxt(file_name, dtype=np.str, delimiter=',', skiprows=1)
    data = tem.astype(np.float)
    return data

def write_to_csv(file_name, data):
    with open(file_name,"w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    print("Done")


def transform_label(data):
    num_of_rows = data.shape[0]
    for each_row in range(num_of_rows):
        if data[each_row,-1] > 3:
            data[each_row,-1] = 0
        else:
            data[each_row,-1] = 1


file_name = 'GData_test_10.csv'
file_name_new = 'GData_test_10_new.csv'
data= loadData(file_name)
num_of_rows = data.shape[0]
transform_label(data)
write_to_csv(file_name_new, data)