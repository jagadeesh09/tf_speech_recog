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


''' Setting up different Variables and initiating seed '''
dir_path = ''    # Path for the pkl files folder
random.seed(338)
int_length = 14   # Number of sets


file_list = os.listdir(dir_path)
file_list = [os.path.join(dir_path, f) for f in file_list ]

''' For loading pickle file continuously '''
def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass


for file_ in file_list:
    with open(file_) as f:
        data = []
        for event in pickleLoader(f):
            b = event
            c = np.reshape(b,(2,192,81))
            data.append(c)
        print(file_)
        with open(file_[:-4]+'1.pkl','wb') as g:
            pickle.dump(data, g, protocol=cPickle.HIGHEST_PROTOCOL)

file_list = [os.path.join(dir_path, f) for f in file_list if f.endswith('1.pkl')]


for i in range(int_len):
    data = []
    labels = []
    class_count = 0
    for file_ in file_list:
        with open(file_) as f:
            for event in pickleLoader(f):
                class_data = event
        num_objects = np.shape(class_data)
        class_count = class_count + 1
        int_class = math.floor(num_objects[0]/14)
        req_data = class_data[i* int_class: (i+1) * int_class]
        cur_labels = [class_count] * (((i+1)* int_class) - (i*class_int))
        labels.append(cur_labels)
        data.append(req_data)
    with open('set'+ str(i)+'.pkl', "wb") as g:
        pickle.dump((data,labels), g, protocol=cPickle.HIGHEST_PROTOCOL)
