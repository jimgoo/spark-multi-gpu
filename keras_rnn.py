from __future__ import print_function

import os
import time
import fire

import numpy as np
np.random.seed(123)

# for LSTM
from keras.models import Sequential
from keras.layers import Input, LSTM, SimpleRNN, GRU, Dense
from keras import optimizers

# for make_parallel
import tensorflow as tf
tf.set_random_seed(123)

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import common

tf.logging.set_verbosity('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def make_parallel(model, device_ids):
    """Make a Keras model parallel.

    The per GPU batch size should be multiplied by `gpu_count`. 
    Any partial batch with length less than the per GPU batch_size 
    will throw an error.

    Source:
    https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py
    """
    gpu_count = len(device_ids)

    # set the CUDA visible devices environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

    if gpu_count <= 1:
        return model

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, 
                        arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # Merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        return Model(input=model.inputs, output=merged)


def get_stop(n_obs, batch_size_per_gpu):
    # Remove last partial batch so that make_parallel doesn't fail (floor!).
    n_batches = int(np.floor(n_obs / float(batch_size_per_gpu)))
    n_obs = n_batches * batch_size_per_gpu
    return n_obs


def train(n_gpus=1, epochs=10, batch_size_per_gpu=128, backend='tensorflow'):

    device_ids = list(range(n_gpus))

    n_features = 29
    n_steps = 60
    hidden_size = 512
    batch_size = batch_size_per_gpu * max(1, n_gpus)
    
    print('n_gpus', n_gpus)
    print('device_ids', device_ids)
    print('batch_size_per_gpu', batch_size_per_gpu)
    print('batch_size_total', batch_size)

    x_train, y_train = common.load_test_data()

    # Remove last partial batch so that make_parallel doesn't fail (floor!).
    # n_batches = int(np.floor(len(x_train) / float(batch_size)))
    # stop = n_batches * batch_size
    # print('stop', stop)
    # x_train = x_train[:stop]
    # y_train = y_train[:stop]

    model = Sequential()
    model.add(SimpleRNN(hidden_size, input_shape=(n_steps, n_features), 
        return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.SGD(lr=0.01)

    if backend == 'tensorflow':
        model = make_parallel(model, device_ids=device_ids)
        model.compile(loss='mse', optimizer=optimizer)
    elif backend == 'mxnet':
        if n_gpus > 0:
            # requires Keras fork: https://github.com/dmlc/keras.git
            context = ['gpu(%d)' % i for i in range(n_gpus)]
        else:
            context = None
        model.compile(loss='mse', optimizer=optimizer, context=context)
    else:
        raise ValueError('Unknown backend: %s' % backend)

    print('-'*50)
    for w in model.get_weights():
        print(w.shape)

    n_pars = sum([np.prod(w.shape) for w in model.get_weights()])
    print('Number of parameters: {:,}'.format(n_pars))

    batch_indices = common.make_batches(len(x_train), batch_size)
    nb_batches = len(batch_indices)

    with open('keras_lstm_x%i_%s.csv' % (n_gpus, backend), 'w+') as f:
        f.write('epoch, epoch_time, loss\n')

        for epoch in range(epochs):
            train_loss = 0.
            epoch_time = time.time()

            for batch_num, (start, stop) in enumerate(batch_indices):
                x_batch = x_train[start:stop]
                y_batch = y_train[start:stop]
                train_loss += model.train_on_batch(x_batch, y_batch)

            #hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
            #train_loss = hist.history['loss'][0]

            train_loss = train_loss/(batch_num+1)
            epoch_time = time.time() - epoch_time
            
            print('epoch: %i, time: %.2f, loss: %.4f' \
                % (epoch, epoch_time, train_loss))

            f.write('%i, %.4f, %.4f\n' % (epoch, epoch_time, train_loss))

        # import keras
        # model.save('/tmp/test-model.h5')
        # model2 = keras.models.load_model('/tmp/test-model.h5')
        # model2.summary()

if __name__ == '__main__':
    fire.Fire()

