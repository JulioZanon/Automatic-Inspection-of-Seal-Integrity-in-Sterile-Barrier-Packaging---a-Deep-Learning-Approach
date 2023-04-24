from custom_lib.user_functions import generate_plots,  create_sumary_logs_by_dataset
import os
import numpy as np

log_folder = 'Report_T4'
print=False
exclusion_string=[]
inclusion_string=['Testing_T4.']



generate_plots(log_folder, print=print, exclusion_string = exclusion_string, inclusion_string= inclusion_string)
create_sumary_logs_by_dataset(log_folder, exclusion_string = exclusion_string, inclusion_string= inclusion_string)

