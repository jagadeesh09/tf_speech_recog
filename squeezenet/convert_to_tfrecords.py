import tensorflow as tf
import cPickle as pickle
import numpy as np
#from object_detection.utils import dataset_util
dir_path = '/run/user/1000/gvfs/smb-share:server=uncannynas,share=uvdata/Audio/Kaggle/train/pkls/'
filename = ['set0.pkl','set1.pkl','set2.pkl','set3.pkl']
file_path = []
''' Creating list of filepaths'''
for file_ in filename:
    file_path.append(dir_path+file_)




def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_to_tfrecord(images,labels,name):
    num_labels = np.shape(labels)
    (num_images,depth,rows,cols) = np.shape(images)
    file_name = "set"+name+".tfrecords"
    writer = tf.python_io.TFRecordWriter(file_name)
    for index in range(num_images):
        image_raw = images[index]
        #image_raw = np.array(image_raw,np.uint8)
        image_raw = np.reshape(image_raw,(192,81,2))
        image_raw = image_raw.astype(np.float32)
        image_raw = image_raw.tostring()
        #print(image_raw.dtype)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
            #'image_raw': _float_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


cnt = 0
for file_ in file_path:
    print("I am working on this file %s" %(file_))
    with open(file_,"rb") as f:
        for event in pickleLoader(f):
            (set_data,label) = event
    counter = 0
    arr_ = []
    larr_ = []
    for data_,lab_ in zip(set_data,label):
        #f = np.asarray(data_)
        #print(f.dtype)
        if(counter == 0):
            arr_ = data_
            larr_ = lab_
            print(np.shape(data_))
        else:
            arr_ = np.append(arr_,data_,axis =0 )
            larr_ = np.append(larr_,lab_,axis=0 )
            print(np.shape(arr_))
            print(np.shape(larr_))
        counter = counter + 1
    convert_to_tfrecord(arr_,larr_,str(cnt))
    cnt = cnt + 1
    '''if(cnt == 0):
        break'''
