import handle_data
from scipy import sparse


# ----------------------------------set parameters--------------------------------------------------------------------
file_name = 'GData_new.csv'
# file_name = 'GData_test_10.csv'
# ----------------------------------start processing--------------------------------------------------------------------
data, label = handle_data.loadData(file_name)
dicstart, diclength = handle_data.group(data)

record_name = 'all_distance.npz'

new_data = handle_data.standarize_PCA_data(data)

distance_matric = handle_data.create_distance_matric(new_data)

sdistance = sparse.csr_matrix(distance_matric)
sparse.save_npz(record_name, sdistance)

# data = sparse.load_npz(record_name)
# data = data.toarray()
# print(data)