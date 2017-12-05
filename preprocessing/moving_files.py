################################################################################
'''                   Author : Jagadeesh Dondeti
                      Date   : December 5th , 2017
                      About  : Code for moving validation dataset into a different folder
                      Competition : Tensorflow Speech Recognition Challenge     '''
################################################################################



############### Importing Libraries ############################################
import os
import shutil
###############################################################################
'''
                txt_file_path : Path for the testing_list.txt file
                dir_path      : Path for train folder of dataset             '''

txt_file_path = '/home/uv/jagadeesh/kaggle/speech/input/train/testing_list.txt'
dir_path = '/home/uv/jagadeesh/kaggle/speech/input/train/'

if not os.path.exists(dir_path+'valid'):
    os.makedirs(dir_path+ 'valid')
file_list = []
with open(txt_file_path,"r") as f:
    file_list = f.read().splitlines()

for each_file in file_list:
    split_name = each_file.split('/')
    if not os.path.exists(dir_path + 'valid/' + split_name[0]):
        os.makedirs(dir_path + 'valid/' + split_name[0])
        shutil.move(dir_path + 'audio/' + each_file, dir_path + 'valid/' + split_name[0])
    else:
        shutil.move(dir_path + 'audio/' + each_file, dir_path + 'valid/' + split_name[0])
