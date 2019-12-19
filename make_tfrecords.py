""" TFRecord generation """
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('src')
import decan_utils as utils
import model_utils
from make_dataset import fetch_filename, load_fasta

MYPATH = os.getcwd()

def convert_to_tfr(x, y, path):
    """
        Each x is a (1,93) int32 array
        Each y is a float64
    """
    print("writing to {}".format(path))
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in range(x.shape[0]):
            x_curr = x[i]
            x_int_list = tf.train.Int64List(value = x_curr[0])
            x_feature = tf.train.Feature(int64_list = x_int_list)

            y_curr = y[i]
            y_float = float(y_curr)
            y_float_list = tf.train.FloatList(value = [y_float])
            y_feature = tf.train.Feature(float_list = y_float_list)

            feature_dict = {'data': x_feature, 'label': y_feature}
            feature_set  = tf.train.Features(feature = feature_dict)
            example      = tf.train.Example(features = feature_set)

            writer.write(example.SerializeToString())
            if i % 1000 == 0:
                print("writing record {}".format(i))

        print("{} records written to {}".format(x.shape[0], path))

def main(protein, fasta='', pdb='', logging=None):
    """TFRecord generation

    Parameters
    ----------
    protein: str,
        Name of protein to collect data.
    fasta : str
        Name of fasta file in ../data/raw (the default is '', which will create an error message if protein name is not given).
    pdb : {str}, optional
        Name of pdb file in ../data/raw (the default is '', which will create an error message if protein name is not given).
    logging : object
        Logging object
    """
    if utils.protein_check(protein, fasta, pdb, logging=logging):
        logging.info('TFRecord generation, data acquisition and cleaning...')

        # acquire raw sequence and reference sequence
        fasta_file, pdb_file = fetch_filename(protein, fasta, pdb, logging=logging)
        sequence, ref_sequence = load_fasta(fasta_filename=fasta_file, protein=protein, logging=logging, pickle=False)

        # sequence translation
        sequence_length = len(sequence[1])
        sequence_onehot, sequence_translated = utils.translate_dict(seq=sequence, seq_length=sequence_length)
        ref_sequence_translated = utils.translate_sequence(seq=ref_sequence)

        # HSW/KL divergence
        HSW = utils.sequence_weights(x=sequence_translated, seq_length=sequence_length)
        DKL = utils.KL_divergence(seqs=sequence_translated, ref_seq=ref_sequence_translated)

        # generate train, validation and test sets
        x_train, x_test, x_val, y_train, y_test, y_val = model_utils.process_dataset(
            sequence_translated=sequence_translated,
            HSW=HSW,
            logging=logging
        )

        # combine train and validation data, Estimator perform split if desired
        combo_x_train = np.concatenate((x_train, x_val))
        combo_y_train = np.concatenate((y_train, y_val))

        # generate TFRecords in the std location
        train_name = 'train_' + protein + '.tfrecords'
        test_name  = 'test_'  + protein + '.tfrecords'
        train_path = os.path.join(utils.data_path(MYPATH, 'processed'), train_name)
        test_path  = os.path.join(utils.data_path(MYPATH, 'processed'), test_name)

        convert_to_tfr(combo_x_train, combo_y_train, train_path)
        convert_to_tfr(x_test, y_test, test_path)

        logging.info('TFRecords generated!')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers. Must provide EITHER protein or fasta/pdb combo.')
    parser.add_argument('--protein', type=str, help='Protein name.')
    parser.add_argument('--fasta', type=str, help='Fasta file name.', default='')
    parser.add_argument('--pdb', type=str, help='PDB file name.', default='')
    parser.add_argument('--logging', type=str, help='Logging option.', default=utils.logger())

    args = vars(parser.parse_args())
    main(**args)
