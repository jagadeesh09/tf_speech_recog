###############################################################################
'''             Author : Jagadeesh Dondeti
                Date   : Nov 30 2017
                About  : Code for creating training and validation sets
                Competition: Tensorflow speech recognition Challenge         '''
###############################################################################

''' Importing required libraries '''

import cPickle as pickle
import os
from random import shuffle
import math
import random
import numpy as np
import random

''' Setting up different Variables and initiating seed '''
dir_path = '/run/user/1000/gvfs/smb-share:server=uncannynas,share=uvdata/Audio/Kaggle/train/pkls/'    # Path for the pkl files folder
random.seed(338)
int_length = 4   # Number of sets

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass
'''
file_list = os.listdir(dir_path)
file_list = [os.path.join(dir_path, f) for f in file_list ]
#print(file_list)




for file_ in file_list:
    print(file_)
    with open(file_) as f:
        data = []
        for event in pickleLoader(f):
            b = event
            c = np.reshape(b,(2,192,81))
            data.append(c)
        print(file_)
        with open(file_[:-4]+'1.pkl','wb') as g:
            pickle.dump(data, g, protocol=pickle.HIGHEST_PROTOCOL)
'''
file_list = os.listdir(dir_path)
file_list = [os.path.join(dir_path, f) for f in file_list if f.endswith('1.pkl')]

with open('outfile.txt', 'w') as fp:
    pickle.dump(file_list, fp)
for i in range(int_length):
    print(" I am currently on %d set" %(i))
    data = []
    labels = []
    class_count = 0
    for file_ in file_list:
        print(file_)
        with open(file_) as f:
            for event in pickleLoader(f):
                class_data = event
        num_objects = np.shape(class_data)
        int_class = math.floor(num_objects[0]/(int_length))
        req_data = class_data[int(i* int_class): int((i+1) * int_class)]
        cur_labels = [class_count] * (int(((i+1)* int_class) - (i*int_class)))
        labels.append(cur_labels)
        data.append(req_data)
        class_count = class_count + 1
    combined_list = list(zip(data,labels))
    random.shuffle(combined_list)
    data,labels = zip(*combined_list)
    with open(dir_path+'set'+ str(i)+'.pkl', "wb") as g:
        pickle.dump((data,labels), g, protocol=pickle.HIGHEST_PROTOCOL)
