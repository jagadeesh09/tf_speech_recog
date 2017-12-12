import tensorflow as tf
import cPickle as pickle
import numpy as np
import squeezenet
import argparse
#key,value = reader.read(file_path)
dir_path = '/home/uv/jagadeesh/kaggle/speech/tf_speech_recog/squeezenet/'
learning_rate = 0.01
batch_size = 128
num_epochs = 2
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    print(filename_queue)
    _, serialized_example = reader.read(filename_queue)
    print(serialized_example)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    print(image)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    print(width)
    depth = tf.cast(features['depth'], tf.int32)
    image_shape = tf.stack([depth,height, width])
    image = tf.reshape(image, image_shape)
    print(np.shape(image))
    print(np.shape(label))
    return image,label

def inputs(batch_size, num_epochs):
    filename = ['set1.tfrecords']
    file_path = []
    ''' Creating list of filepaths'''
    for file_ in filename:
        file_path.append(dir_path+file_)
    print(filename)
    print(file_path)
    file_path = dir_path + 'set1.tfrecords'
    filename_queue = tf.train.string_input_producer([file_path], num_epochs=num_epochs)
    image,label = read_and_decode(filename_queue)
    print(np.shape(image))
    print(np.shape(label))
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    return images, sparse_labels

def do_training():
    with tf.Graph().as_default():
        images,labels = inputs(batch_size,num_epochs)
        logits = squeezenet.inference(images)
        loss = squeezenet.loss(logits,labels)
        train_op = squeezenet.training(loss, learning_rate)
        init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess = tf.Session()
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
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
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
