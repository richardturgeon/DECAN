
from __future__ import print_function
 
import tensorflow as tf
#tf.enable_eager_execution() 
 
def main():
 
    def extract_fn(data_record):
        features = {
            'data': tf.FixedLenFeature([93], dtype=tf.int64),
            'label': tf.FixedLenFeature([], dtype=tf.float32)
        }
        sample = tf.parse_single_example(data_record, features)
        x = sample['data']
        y = sample['label']
#       x = tf.reshape(x, shape=[93])
        return x, y
 
    dataset = tf.data.TFRecordDataset(['/vol/ml/turgeon/DECAN/data/processed/train_cadherin.tfrecords'])
    dataset = dataset.map(extract_fn)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
 
    with tf.Session() as sess:
        x, y = sess.run(next_element)
        print(x, y)
 
 
if __name__ == '__main__':
    main()
 
