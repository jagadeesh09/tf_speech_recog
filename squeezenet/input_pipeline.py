import tensorflow as tf
import cPickle as pickle
import numpy as np
import squeezenet
import argparse
import pdb
import time
#key,value = reader.read(file_path)
dir_path = '/home/uv/jagadeesh/kaggle/speech/tf_speech_recog/squeezenet/'
learning_rate = 0.01
batch_size = 128
num_epochs = 200
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'image_raw': tf.FixedLenFeature([], tf.string),
          #'image_raw': tf.FixedLenFeature([], tf.float64),
          'label': tf.FixedLenFeature([], tf.int64),
      })
    #pdb.set_trace()
    #image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.decode_raw(img_features['image_raw'], tf.float32)
    label = tf.cast(img_features['label'], tf.int32)
    height = tf.cast(img_features['height'], tf.int32)
    width = tf.cast(img_features['width'], tf.int32)
    depth = tf.cast(img_features['depth'], tf.int32)
    image_shape = [192, 81, 2]
    image = tf.reshape(image,image_shape)
    image.set_shape(image_shape)
    return image,label
def modified_read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    print(filename_queue)
    _, serialized_example = reader.read(filename_queue)
    print(serialized_example)
    img_features = tf.parse_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'image_raw': tf.FixedLenFeature([], tf.string),
          #'image_raw': tf.FixedLenFeature([], tf.float64),
          'label': tf.FixedLenFeature([], tf.int64),
      })
    print(img_features["height"])
    #pdb.set_trace()
    #image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.decode_raw(img_features['image_raw'], tf.float32)
    print(img_features['label'])
    print(image)
    label = tf.cast(img_features['label'], tf.int32)
    height = tf.cast(img_features['height'], tf.int32)
    width = tf.cast(img_features['width'], tf.int32)
    print(width)
    depth = tf.cast(img_features['depth'], tf.int32)
    image_shape = tf.stack([depth,height, width])
    image = tf.reshape(image, image_shape)
    image.set_shape(image_shape)
    print(np.shape(image))
    print(np.shape(label))
    return image,label
def simple_read_and_decode(filename):
    record_iterator = tf.python_io.tf_record_iterator(path=filename)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])
        print(height)
        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])

        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])


        img_1d = np.fromstring(img_string, dtype=np.float32)
        print(np.shape(img_1d))
        reconstructed_img = img_1d.reshape((height, width, -1))

def inputs(batch_size, num_epochs):
    filename = ['set0.tfrecords','set1.tfrecords','set2.tfrecords','set3.tfrecords']
    file_path = []
    ''' Creating list of filepaths'''
    for file_ in filename:
        file_path.append(dir_path+file_)
    print(filename)
    print(file_path)
    #file_path = dir_path + 'set1.tfrecords'
    print(file_path)
    filename_queue = tf.train.string_input_producer(file_path, num_epochs=num_epochs)
    image,label = read_and_decode(filename_queue)
    #image,label = simple_read_and_decode(file_path)
    print(np.shape(image))
    print(np.shape(label))
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    return images, sparse_labels

def do_training():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            images,labels = inputs(batch_size,num_epochs)
            logits = squeezenet.inference(images,batch_size)
            loss = squeezenet.loss(logits,labels)
            train_op = squeezenet.training(loss, learning_rate)
            init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                step =0
                while not coord.should_stop():
                    time_ = time.time()
                    _, loss_value = sess.run([train_op, loss])
                    duration = time.time() - time_
                    if(step % 50 == 0):
                        print("Step: %d Loss : %.3f Duration: %.3f seconds" %(step,loss_value,duration))
                    step = step + 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (num_epochs, step))
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()
def main(unused_argv):
    do_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate')
    tf.app.run()
