# 00_SetUp_Project_First_Use Rev:01.00
"""
Scrip create the required folder structure to run the framework.
"""
import os
path = os.path.join(os.getcwd(), 'config')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'datasets')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'logs')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'models')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'models\\00_search_files')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'models\\01_trained')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'models\\02_test')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'notation')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'notation\\COCO_rev3')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)
path = os.path.join(os.getcwd(), 'config')
if not os.path.isdir(path): os.makedirs(path, exist_ok=False)