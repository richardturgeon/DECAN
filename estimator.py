import argparse
import glob
import numpy as np
import os
import sys
import tensorflow as tf
import keras
from keras import backend as K


#tf.enable_eager_execution()

sys.path.append('src')
import attention_decoder
import decan_utils as utils
from   decan import load_model

MYPATH = os.getcwd()

"""
from google.protobuf import text_format
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
"""

SHUFFLE_BUFFER = 1500
BATCH_SIZE = 128 # ???????????????????????????????????????????????/

##______________________________________________________________________________
def main(protein, train=False, eval=False, data_dir=None, model_dir=None, logging=None):

    config = utils.load_config()

    train_batch_size = BATCH_SIZE
    test_batch_size = BATCH_SIZE
    train_steps = 100
    train_steps = 1000

    """
    params = {
        'hidden_units': [128],
        'n_classes': NUM_CLASSES,
        'dropout': 0.1,
        'lr': 2e-2,
        }
    """

    ## build estimator
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=config)

    ## train model
    if train:
        _input_fn = lambda: input_fn(data_dir, train_batch_size, is_training=True)
        model.train(input_fn=_input_fn, steps=train_steps)
        print('==> Trained')

    ## evaluate model
    if eval:
        _eval_input_fn = lambda: input_fn(data_dir, test_batch_size, is_training=False)
        eval_result = model.evaluate(input_fn=_eval_input_fn)
        print('\n')
        print('==> Test set accuracy: {accuracy:0.3f}'.format(**eval_result))
        print('\n')


##______________________________________________________________________________
def input_fn(data_dir, batch_size, is_training=True, prep_style='minimal', num_parallel_reads=0):
    """   """

    feature_map = {
        'data':     tf.FixedLenFeature([93],  dtype=tf.int64),
        'label':    tf.FixedLenFeature([], dtype=tf.float32)
    }

    filenames = get_filenames(data_dir, is_training, fmt='tfrecords')
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
        # Shuffle the input files
        dataset = dataset.shuffle(buffer_size=len(filenames))

    # Convert to individual records
    if num_parallel_reads >= 1:
        dataset = dataset.flat_map(lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=num_parallel_reads))
    else:
        dataset = dataset.flat_map(tf.data.TFRecordDataset)

#    if cache_data:
#        dataset = dataset.take(1).cache().repeat()

    def parse_record_fn(raw_record, is_training):
        return parse_record(raw_record, is_training=is_training, feature_map=feature_map, prep_style=prep_style)

    return process_record_dataset(dataset, is_training, batch_size, SHUFFLE_BUFFER, parse_record_fn)


##_____________________________________________________________________________
def get_filenames(data_dir, is_training=True, fmt='tfrecords'):
    """Return filenames for dataset."""
    if is_training:
        tfrecords = glob.glob(os.path.join(data_dir, 'train*.%s' % fmt))
    else:
        tfrecords = glob.glob(os.path.join(data_dir, 'test*.%s' % fmt))
    tfrecords.sort()
    return tfrecords


##_____________________________________________________________________________
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_batches=1):
    """Given a Dataset with raw records, return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """


    num_epochs = 100 # ???????????????????????????????????????????????????????????????????????????????


    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffle the records. Note that we shuffle before repeating to ensure
        # that the shuffling respects epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # If we are training over multiple epochs before evaluating, repeat the
    # dataset for the appropriate number of epochs.
    dataset = dataset.repeat(num_epochs)

    # Parse the raw records into images and labels. Testing has shown that setting
    # num_parallel_batches > 1 produces no improvement in throughput, since
    # batch_size is almost always much greater than the number of CPU cores.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training=is_training),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=True))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset


##_____________________________________________________________________________
def parse_record(raw_record, feature_map, is_training=True, prep_style='minimal'):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.

    Returns:
      Tuple with processed image tensor and one-hot-encoded label tensor.
    """
    prep_style = prep_style.lower()
    assert prep_style == 'minimal'

    record_features = tf.parse_single_example(raw_record, feature_map)
    data  = record_features['data']
#   data  = tf.cast(data, tf.float32)            # ??????????????????????????????????????
    label = record_features['label']
    label = tf.cast(label, tf.int32)             # ??????????????????????????????????????

    features = {'x' : data}

    return features, label

##______________________________________________________________________________
def model_fn(features, labels, mode, params):
    """
    """

    # model_list = load_model(93, .3, 21, args['logging'])   # works! ???????????????????????????????
    estim_spec = None
    x = features['x']

    # ??????????????????????????????????????????????????????????????????
    SEQ_LEN = 93
    params = {}
    params['vocab_size']    = 21
    params['dropout']       = .3

    CONFIG = utils.load_config()
    # ??????????????????????????????????????????????????????????????????

    remap1 = tf.reshape(x,[tf.shape(x)[0], SEQ_LEN, 1])  # ??????????????????????????????????????????????
    remap2 = tf.reshape(x,[tf.shape(x)[0], SEQ_LEN])  # ??????????????????????????????????????????????

    if(mode==tf.estimator.ModeKeys.PREDICT or mode==tf.estimator.ModeKeys.EVAL):
        dropout_mode=False
    else:
        dropout_mode=True

    dropout_mode=True ## TODO: why do i need this for evaluation?

    dropout = params['dropout']
    vocab_size = params['vocab_size']

    # define Residue RNN
#   with tf.device("/job:localhost/replica:0/task:0/device:XLA_CPU:0"):   # LSTM is GPU only ???????????????????????
    with tf.device("/job:localhost/replica:0/task:0/device:GPU:0"):
        with tf.variable_scope("model_fn", use_resource=True):
#           net = keras.layers.Input(shape=(93,), dtype='int32', name='residue_in', tensor=remap1
            net = keras.layers.Input(shape=(93,), dtype='int32', name='residue_in', tensor=remap2)
#           net = keras.layers.Input(shape=(93,), dtype='int32', name='residue_in')  # ?????????????????????????????
            net_input = net
            net = keras.layers.Embedding(SEQ_LEN, vocab_size, mask_zero=True, trainable=True)(net)

            if dropout > 0.0:
                net = keras.layers.Dropout(dropout)(net)

            net = keras.layers.Bidirectional(
                keras.layers.LSTM(vocab_size, return_sequences = True),
                merge_mode = 'sum',
                name = 'BLSTM1'
            )(net)

            if dropout > 0.0:
                net = keras.layers.Dropout(dropout)(net)
            
            attn = attention_decoder.AttentionDecoder(SEQ_LEN, vocab_size, name="AttnWrapper")(net)
            net  = keras.layers.multiply([net, attn], name = 'Merge')
            net  = keras.layers.core.Lambda(lambda x:K.sum(x,axis=1), output_shape=lambda x:(x[0], x[2]), name='Sum')(net)

            logits = keras.layers.Dense(1, activation=None)(net)     # ???????????????????????

            # ???????????????? modelRes not used, just to get the picture!  ??????????????
            modelRes = keras.models.Model(net_input, net)
            modelRes.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', 'mse'])
            modelRes.summary()
            # ???????????????? not used ??????????????

            # Compute predictions.
            predicted_classes = tf.argmax(logits, 1)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits,
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            # Compute loss.
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            # Compute evaluation metrics.
            accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=predicted_classes,
                name='acc_op'
            )

            metrics = {'accuracy': accuracy}
            tf.summary.scalar('accuracy', accuracy[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    eval_metric_ops=metrics
                )

            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN

            optimizer = tf.train.AdamOptimizer(learning_rate=.02)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            estim_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return estim_spec


##______________________________________________________________________________
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--protein',
                        type=str,
                        required=True,
                        help='Protein name.')

    parser.add_argument('--logging',
                        type=str,
                        help='Logging option.',
                        default=utils.logger())

    # candle-like                    
    parser.add_argument('--model_dir',
                        default=os.path.join(MYPATH, 'model_dir'),
                        type=str,
                        help='tensorflow model_dir.')

    parser.add_argument('--data_dir',
                        default=os.path.join(utils.data_path(MYPATH, data_dir='processed')),
                        type=str,
                        help='tensorflow model_dir.')

    parser.add_argument('--train',
                        default=False,
                        action='store_true',
                        help='Train the model.')

    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='Evaluate the model.')

    args = vars(parser.parse_args())
    main(**args)

