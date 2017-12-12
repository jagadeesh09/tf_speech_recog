import tensorflow as tf
import cPickle as pickle
import numpy as np

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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_to_tfrecord(images,labels,name):
    num_labels = np.shape(labels)
    (num_images,depth,rows,cols) = np.shape(images)
    file_name = "set1.tfrecords"
    writer = tf.python_io.TFRecordWriter(file_name)
    for index in range(num_images):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


cnt = 0
for file_ in file_path:
    with open(file_,"rb") as f:
        for event in pickleLoader(f):
            (set_data,label) = event
    print(np.shape(set_data))
    print(np.shape(label))
    counter = 0
    arr_ = []
    larr_ = []
    for data_,lab_ in zip(set_data,label):
        if(counter == 0):
            arr_ = data_
            larr_ = lab_
            print(np.shape(data_))
        elif(counter > 40):
            arr_ = np.append(arr_,data_,axis =0 )
            larr_ = np.append(larr_,lab_,axis=0 )
            print(np.shape(arr_))
            print(np.shape(larr_))
        counter = counter + 1
    convert_to_tfrecord(arr_,larr_,"jag")
    if(cnt == 0):
        break
