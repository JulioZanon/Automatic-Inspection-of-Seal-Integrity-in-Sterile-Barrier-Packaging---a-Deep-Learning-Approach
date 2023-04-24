# 02b_Test_Model Rev:01.00

#imports
from custom_lib.json_function import read_from_json
from custom_lib.model_tune import tune_model
from custom_lib.model_test import test_model
from custom_lib.model_build_funtions import check_GPUs_availibility
from custom_lib.user_functions import check_trained_model_files
from custom_lib.json_function import dictionary_to_array
import os


exclusion_string=[]
inclusion_string=[]
log_name= 'Testing_T'
v_show_plots = False
GPU = 1  #Set to None to ignore GPU selection
log_name = log_name + '_' + str(GPU)
GPU_count = check_GPUs_availibility(GPU)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
GPU_count = check_GPUs_availibility(GPU)
dataset_path = 'middle_seal_breach\\Test_EG2'
dataset_splits = ['Testing_T1', 'Testing_T2', 'Testing_T3', 'Testing_T4']
show_plots = False


array_search_paths_h5, array_search_paths_json = check_trained_model_files(exclusion_string, inclusion_string)

for model_file_h5, model_file_json in zip(array_search_paths_h5, array_search_paths_json):
    model_info = read_from_json(model_file_json)
    for dataset_split in dataset_splits:
        metrics, summary_log = test_model(model=model_file_h5, model_info=model_info,
                                          dataset_info_name = dataset_path,
                                          dataset_test_split = dataset_split,
                                          log_name=log_name,
                                          show_plots=show_plots,
                                          test_Batch_size=8)

