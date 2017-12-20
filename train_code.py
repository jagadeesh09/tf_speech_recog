import numpy as np
import tensorflow as tf
import os
import re
from glob import glob
import librosa
import librosa.display as display
from scipy.io import wavfile

data_dir = '/home/uv/jagadeesh/kaggle/speech/input'

class_labels = 'bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven sheila six stop three tree two up wow yes zero silence unknown'.split()

id2name = {i: name for i, name in enumerate(class_labels)}
name2id = {name: i for i, name in id2name.items()}
cnt = 0
def load_data(data_dir):
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as f:
            validation_files = f.readlines()

    valset = set()

    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible_labels = set(class_labels)
    train, val = [], []

    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible_labels:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)
    return train ,val
'''           Data Generator     '''
def data_generator(data,params, mode):
    def check_track (clip,sr):
        max_length = (sr*1)
        #print("I came insode the track")
        length_of_clip = len(clip)
        #Checking the clip for dubious values
        if (np.isnan(np.amin(clip)) or np.isnan(np.amax(clip))):
            return np.zeros(max_length)
        #Checking if length of the clip is 2s or not
        if (length_of_clip > max_length):
            #print "\n\tThe length of the clip is larger than 1s. Shortening to 1s"
            return clip[0:max_length]
        elif (length_of_clip < max_length):
            #print "\n\tThe length of the clip is smaller than 1s. Appending to 1s"
            diff = max_length - length_of_clip
            clip = np.hstack ([clip,np.zeros(diff)])
            return clip
        else:
            #print "\n\tThe clip is perfect!"
            return clip
    def feature_extractor(track):
        #Loading the audio track
        clip,sr = librosa.core.load(track,sr=8000)
        song = check_track(clip,sr)

        #Exctracting Features
        '''
        Feature Specification:
            Rate: 8000 Hz
            Scale Type: Mel
            Audio Track Length: 2 s
            Frame Size: 20 ms
            STFT Hop Length: 100 (System Default:512)
                            Audio Track Length
            (    Hop Length: ---------------------- )
                                Frame Size
            Frequency Bins: 128
            Number of MFCC: 64
            Number of Deltas: 2
        '''
        #MelSpectogram
        S = librosa.feature.melspectrogram(song, sr=sr,hop_length = 100,n_mels=192)
        log_S = librosa.logamplitude(S,ref_power = np.max)
        #display.specshow(log_S, y_axis='log')
        #plt.show()
        #MFCC and Deltas
        mfcc1 = librosa.feature.mfcc(y=song, sr=8000, S=log_S, n_mfcc=64)
        mfcc2 = librosa.feature.delta(mfcc1)
        mfcc3 = librosa.feature.delta(mfcc1,order=2)
        mfcc = np.vstack ([mfcc1,mfcc2,mfcc3])
        #librosa.display.specshow(mfcc, y_axis='log')
        #plt.show()
        #Array Manipulations
        log_S = log_S[:,:,np.newaxis]				#2-D to 3-D array
        log_S = log_S[:,:,:,np.newaxis]              #3-D to 4-D array
        log_S = np.transpose(log_S,(3,2,0,1))
        #print "\n\tShape of the Mel Spectorgram:",np.shape(log_S)
        mfcc = mfcc[:,:,np.newaxis]				#2-D to 3-D array
        mfcc = mfcc[:,:,:,np.newaxis]              #3-D to 4-D array
        mfcc = np.transpose(mfcc,(3,2,0,1))
        #print "\n\tShape of the MFCC Deltas:",np.shape(mfcc)
        picture = np.stack([log_S,mfcc],axis=1)
        #print "\n\tShape of picture",np.shape(picture)
        return picture
    ################################################################################
    def generator():
        if(mode == 'train'):
            np.random.shuffle(data)
        print("I am starting generating the data")
        print(np.shape(data))
        #cnt = 0
        #data = data[:3000]
        for(label_id, u_id, entry) in data:
            #cnt = cnt + 1
            #print(cnt)
            try:
                _, wav = wavfile.read(entry)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0
                    yield dict(
                        label=np.int32(label_id),
                        image=wav[beg: beg + L],
                    )
            except Exception as err:
                print(err, label_id, u_id, entry)
        print("I am coming out of the data generation")

    return generator

################################################################################

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

def inference(features,batch_size,num_classes,is_training):
    #print("mode")
    #print(mode)
    #print("count %d " %(cnt))
    #cnt = cnt + 1
    width = 81# input witdth
    height = 192# input height
    channels = 2#input channels
    #num_classes = 30# Number of classess
    print(" I came inside the inference")
    ''' Input Layer'''
    #input_layer = tf.reshape(features,[-1,height,width,channels])
    input_layer = features
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
    dropout = tf.layers.dropout(fire_9, rate= 0.5 if is_training else 1.0)

    ''' Layer - 2 '''
    conv10 = tf.layers.conv2d(inputs=dropout, filters=num_classes,
                    kernel_size = [1,1],
                    activation = tf.nn.relu)
    output = tf.layers.average_pooling2d(inputs=conv10, pool_size= [5,15], strides = (1,1))

    logits = tf.layers.dense(inputs=output, units=num_classes)
    logits = tf.reshape(logits,(batch_size,num_classes))
    return logits

from tensorflow.contrib import signal
def model_handler(features, labels, mode, params, config):
    # Im really like to use make_template instead of variable_scopes and re-usage
    extractor = tf.make_template(
        'extractor', inference,
        create_scope_now_=True,
    )
    image = features['image']
    # we want to compute spectograms by means of short time fourier transform:
    specgram = signal.stft(
        image,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))

    x = tf.stack([amp, phase], axis=3) # shape is [bs, time, freq_bins, 2]
    x = tf.to_float(x)
    # wav is a waveform signal with shape (16000, )
    #image = features['image']
    # we want to have float32, not float64

    logits = extractor(x, params.batch_size,params.num_classes, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                acc=(acc, acc_op),
            )
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
            'sample': features['sample'], # it's a hack for simplicity
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)
def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=model_handler,
        config=config,
        params=hparams,
    )
params=dict(
    seed=2018,
    batch_size=64,
    keep_prob=0.5,
    learning_rate=1e-3,
    clip_gradients=15.0,
    use_batch_norm=True,
    num_classes=len(class_labels),
)

hparams = tf.contrib.training.HParams(**params)
OUTDIR = './model-2'
directory = os.path.join(OUTDIR, 'eval')
if not os.path.exists(directory):
    os.makedirs(directory)
#os.makedirs(os.path.join(OUTDIR, 'eval'))
model_dir = OUTDIR

run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)

trainset, valset = load_data(data_dir)

# it's a magic function :)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

train_input_fn = generator_input_fn(
    x=data_generator(trainset, hparams, 'train'),
    target_key='label',  # you could leave target_key in features, so labels in model_handler will be empty
    batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
)

val_input_fn = generator_input_fn(
    x=data_generator(valset, hparams, 'val'),
    target_key='label',
    batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
)


def _create_my_experiment(run_config, hparams):
    exp = tf.contrib.learn.Experiment(
        estimator=create_model(config=run_config, hparams=hparams),
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        train_steps=150000, # just randomly selected params
        eval_steps=200,  # read source code for steps-epochs ariphmetics
        train_steps_per_iteration=1000,
    )
    return exp

tf.contrib.learn.learn_runner.run(
    experiment_fn=_create_my_experiment,
    run_config=run_config,
    schedule="continuous_train_and_eval",
    hparams=hparams)
