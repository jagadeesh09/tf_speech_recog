################################################################################
#     Author : Dondeti Jagadeesh                                               #
#     Date : Nov 30, 2017                                                      #
#     Model : Squeeze Net                                                      #
#                                                                              #
################################################################################

import tensorflow as tf


model_dir = # Directory to store checkpoints
def fire_module(input, squeeze_depth, expand_depth):
    '''Squeezing Layer'''
    squeeze_output = tf.layers.conv2d(inputs = input, filters = squeeze_depth,
                            kernel_size = [1,1], stride = 1)
    ''' Expand Layer'''
    expand_layer1x1 = tf.layers.conv2d(inputs = squeeze_output, filters =
                            expand_depth, kernel_size = [1,1], stride = 1)
    expand_layer3x3 = tf.layers.conv2d(inputs = squeeze_output, filters =
                            expand_depth, kernel_size = [3,3], stride = 1)

    fire_module_output = tf.concat([expand_layer1x1, expandlayer3x3], axis = 0)

    return fire_module_output



def squeezenet(input,labels, is_training = True):
    width = # input witdth
    height = # input height
    channels = #input channels
    num_classes = # Number of classess
    ''' Input Layer'''
    input_layer = tf.reshape(input,[-1,width,height,channels])

    ''' Layer - 1 '''
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=96,
                    kernel_size = [7,7], stride = 2, padding = "same",
                    activation = tf.nn.relu)

    ''' Pooling Layer - 1 '''
    max_pool_1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [3,3], stride = 2)

    ''' Fire Module -2 , 3, 4 '''

    fire_2 = fire_module(max_pool_1, 16, 64)

    fire_3 = fire_module(fire_2 , 16, 64)

    fire_4 = fire_module(fire_3, 32, 128)

    ''' Pooling Layer - 2 '''
    max_pool_2 = tf.max_pooling2d(inputs = fire_4, pool_size = [3,3], stride = 2)

    ''' Fire Module - 5, 6, 7'''

    fire_5 = fire_module(max_pool_2, 32, 128)

    fire_6 = fire_module(fire_5, 48, 192)

    fire_7 = fire_module(fire_6, 48, 192)

    fire_8 = fire_module(fire_7, 64, 256)

    ''' Pooling Layer - 3'''

    max_pool_3 = tf.layers.max_pooling2d(inputs = fire_8, pool_size = [3,3], stride = 2)

    fire_9 = fire_module(max_pool_3, 64, 256)

    ''' Drop out layer '''
    dropout = tf.layers.dropout(fire_9, keep_prob = 0.5,training=mode == tf.estimator.ModeKeys.TRAIN)

    ''' Layer - 2 '''
    conv10 = tf.layers.conv2d(inputs=dropout, filters=num_classes,
                    kernel_size = [1,1], stride = 1,
                    activation = tf.nn.relu)
    output = tf.layers.average_pooling2d(inputs=conv10, pool_size= [13,13], stride = 1)

    logits = tf.layers.dense(inputs=output, units=num_classes)

    predictions = {
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
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    speech_classifier = tf.estimator.Estimator(
        model_fn=squeezenet, model_dir=model_dir)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,
            batch_size=100, num_epochs=None, shuffle=True)
    results = speech_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
    print(results)


if __name__ == "__main__":
    main()
