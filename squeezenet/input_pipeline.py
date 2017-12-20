import tensorflow as tf
import cPickle as pickle
import numpy as np
import squeezenet
import argparse
import pdb
import time
from datetime import datetime
#key,value = reader.read(file_path)
dir_path = '/home/uv/jagadeesh/kaggle/speech/tf_speech_recog/squeezenet/'
learning_rate = 0.001
batch_size = 64
num_epochs = 300
num_classes = 30




tf.logging.set_verbosity(tf.logging.INFO)
num_classes = 30
batch_size = 64
model_dir =  'check_points/'# Directory to store checkpoints
def fire_module(input, squeeze_depth, expand_depth):
    '''Squeezing Layer'''
    #print("input_shape")
    #print(input.get_shape())
    squeeze_output = tf.layers.conv2d(inputs = input, filters = squeeze_depth,
                            kernel_size = [1,1])
    #print("squeeze output shape")
    #print(squeeze_output.get_shape())
    ''' Expand Layer'''
    expand_layer1x1 = tf.layers.conv2d(inputs = squeeze_output, filters =
                            expand_depth, kernel_size = [1,1])
    #print("expand1 output shape")
    #print(expand_layer1x1.get_shape())
    expand_layer3x3 = tf.layers.conv2d(inputs = squeeze_output, filters =
                            expand_depth, kernel_size = [3,3],padding="same")
    #print("expand3 output shape")
    #print(expand_layer3x3.get_shape())
    fire_module_output = tf.concat([expand_layer1x1, expand_layer3x3], axis = 3)

    return fire_module_output



def inference(features,batch_size):
    #print("mode")
    #print(mode)
    width = 81# input witdth
    height = 192# input height
    channels = 2#input channels
    #num_classes = 30# Number of classess
    ''' Input Layer'''
    input_layer = tf.reshape(features,[-1,width,height,channels])

    ''' Layer - 1 '''
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96,
                    kernel_size = [7,7], strides = (2,2), padding = "same",
                    activation = tf.nn.relu)

    ''' Pooling Layer - 1 '''
    max_pool_1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [3,3], strides = (2,2))

    ''' Fire Module -2 , 3, 4 '''

    fire_2 = fire_module(max_pool_1, 16, 64)

    fire_3 = fire_module(fire_2 , 16, 64)

    fire_4 = fire_module(fire_3, 32, 128)

    ''' Pooling Layer - 2 '''
    max_pool_2 = tf.layers.max_pooling2d(inputs = fire_4, pool_size = [3,3], strides = (2,2))

    ''' Fire Module - 5, 6, 7'''

    fire_5 = fire_module(max_pool_2, 32, 128)

    fire_6 = fire_module(fire_5, 48, 192)

    fire_7 = fire_module(fire_6, 48, 192)

    fire_8 = fire_module(fire_7, 64, 256)

    ''' Pooling Layer - 3'''

    max_pool_3 = tf.layers.max_pooling2d(inputs = fire_8, pool_size = [3,3], strides = (2,2))

    fire_9 = fire_module(max_pool_3, 64, 256)

    ''' Drop out layer '''
    #dropout = tf.layers.dropout(fire_9, rate= 0.5,training=(mode == tf.estimator.ModeKeys.TRAIN))
    dropout = tf.layers.dropout(fire_9, rate= 0.5)

    ''' Layer - 2 '''
    conv10 = tf.layers.conv2d(inputs=dropout, filters=num_classes,
                    kernel_size = [1,1],
                    activation = tf.nn.relu)
    output = tf.layers.average_pooling2d(inputs=conv10, pool_size= [4,11], strides = (1,1))

    logits = tf.layers.dense(inputs=output, units=num_classes)
    logits = tf.reshape(logits,(batch_size,num_classes))
    return logits
    """predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
    loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)"""
def loss(logits, labels):
    one_vector = tf.ones([batch_size],dtype = tf.int32)
    lab = tf.subtract(labels,one_vector)
    onehot_labels = tf.one_hot(indices=tf.cast(lab, tf.int32), depth=num_classes)
    #print("groundtruth")
    #print(labels.eval())
    #labels = tf.Print(labels, [labels], message="This is a: ")
    #b = tf.add(labels, labels).eval()
    #b.eval()
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
    #print("predicted labels")
    '''print(sess.run(logits))
    print("One hot labels")
    print(sess.run(onehot_labels))'''
    loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)
    #loss = tf.reduce_mean(
            #tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(onehot_labels, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("Accuracy",acc)
    return loss
def training(loss,learning_rate,global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step)
    return train_op
def accuracy(logits,labels):
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    print(correct_prediction)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("I am failing inside accuracy")
    return acc



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
    filename_queue = tf.train.string_input_producer(file_path, num_epochs=num_epochs)
    image,label = read_and_decode(filename_queue)
    #image,label = simple_read_and_decode(file_path)
    print(np.shape(image))
    print(np.shape(label))
    images, sparse_labels = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size)#, min_after_dequeue=1000)
    return images, sparse_labels
def train():

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()


        with tf.device('/cpu:0'):
            images, labels = inputs(batch_size,num_epochs)


        logits = squeezenet.inference(images,batch_size)
        loss = squeezenet.loss(logits,labels)
        train_op = squeezenet.training(loss, learning_rate,global_step)

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                log_frequency = 50
                batch_size = 64
                if self._step % log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = log_frequency * batch_size / duration
                    sec_per_batch = float(duration / log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                            examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir="checkpoint/",
                hooks=[tf.train.StopAtStepHook(last_step=20000),
                tf.train.NanTensorHook(loss),
                _LoggerHook()],
                config=tf.ConfigProto(
                log_device_placement=False)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

def do_training():
    #with tf.Graph().as_default():
    with tf.Session() as sess:
        global_step = tf.train.get_or_create_global_step()
        images,labels = inputs(batch_size,num_epochs)
        logits = inference(images,batch_size)
        loss_ = loss(logits,labels)
        #accuracy = squeezenet.accuracy(logits,labels)
        tf.summary.scalar('cross_entropy',loss_)
        #tf.summary.scalar('Accuracy',accuracy)

        tf.summary.tensor_summary('Labels',labels)
        #tf.summary.histogram("weights1_1", filter1_1)
        train_op = training(loss_, learning_rate,global_step)
        init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        writer = tf.summary.FileWriter("output", sess.graph)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #tf.summary.image('images',images[1])
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        ckpt = tf.train.get_checkpoint_state('speech_model/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        #writer.close()
        merged_summary = tf.summary.merge_all()
        #saver.restore(sess,'speech_model/model-4601')
        try:
            step =0
            while not coord.should_stop():
                time_ = time.time()
                print("groundtruth")
                print(sess.run(labels))
                one_vector = tf.ones([batch_size],tf.int32)
                lab = tf.subtract(labels , one_vector)
                print("modified labels")
                print(sess.run(lab))
                onehot_labels = tf.one_hot(indices=tf.cast(lab, tf.int32), depth=num_classes)
                print("predicted labels")
                print(sess.run(tf.argmax(logits,axis=1)))
                print("One hot labels")
                print(sess.run(onehot_labels))
                loss_value,_,summary = sess.run([loss_,train_op,merged_summary])
                #tf.summary.scalar('cross_entropy',loss_value)
                duration = time.time() - time_

                #s = sess.run(merged_summary)
                writer.add_summary(summary,step)
                if(step % 50 == 0):
                    print("Step: %d Loss : %.3f Duration: %.3f seconds" %(step,loss_value,duration))
                    saver.save(sess,'speech_model/model',global_step=tf.train.get_global_step(),write_meta_graph=False)
                step = step + 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (num_epochs, step))
        finally:
            coord.request_stop()
        #saver.save(sess,'speech_model/',global_step=1000)
        writer.close()
        coord.join(threads)
        sess.close()
def main(unused_argv):
    do_training()
    #train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate')
    tf.app.run()
