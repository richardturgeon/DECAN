""" Data Acquisition and Processing """
import os
import os.path as op
import sys
sys.path.append('src')
import decan_utils as utils

from Bio import AlignIO, PDB, Seq, SeqIO
import numpy as np
import pandas as pd
import h5py
import _pickle as cPickle

MYPATH = os.getcwd()


def fetch_filename(protein, fasta, pdb, logging=None):
    """
    Given a protein name or a fasta/pdb pair, retrieves a filepath for both FASTA and PDB files.

    Parameters
    ----------
    protein: str,
        Name of protein to collect data.
    fasta : str
        Name of fasta file in ../data/raw (the default is '', which will create an error message if protein name is not given).
    pdb : str, optional
        Name of pdb file in ../data/raw (the default is '', which will create an error message if protein name is not given).
    logging : object
        Logging object
    Returns
    -------
    fasta_filename and pdb_filename, filepaths for the FASTA and PDB files in the data directory specified ('raw')
    """
    CONFIG = utils.load_config()
    protein_dict = CONFIG['PROTEIN_DICT']
    if protein in protein_dict and (fasta=='' or pdb==''):
        logging.info("Using protein {} to find data files.".format(protein))
        fasta_filename = op.join(utils.data_path(MYPATH, 'raw'), protein_dict[protein]['fasta'] + '.fasta')
        pdb_filename = op.join(utils.data_path(MYPATH, 'raw'), protein_dict[protein]['pdb'] + '.pdb')

    elif (fasta!='' and pdb!=''):
        fasta_filename = op.join(utils.data_path(MYPATH, 'raw'), fasta + '.fasta')
        pdb_filename = op.join(utils.data_path(MYPATH, 'raw'), pdb + '.pdb')

    else:
        logging.error("Invalid protein name or fasta/pdb filenames")
    return fasta_filename, pdb_filename


def load_fasta(fasta_filename, protein, logging=None, pickle=False):
    """
    Loads and processes FASTA file

    Parameters
    ----------
    fasta_filename : str
        Filepath for FASTA file.
    protein: str,
        Name of protein to collect data.
    logging : object
        Logging object

    Returns
    -------
    [[protein-sequence], [reference-sequence]] is returned
    If pickle=True, the aforementioned sequence is stored in ../data/processed/{protein}_seq.pkl  
    No returns, but saves protein sequences and a reference sequence in ../data/processed/{protein}_seq.pkl
    """
    CONFIG = utils.load_config()
    REF_SEQ_DICT = CONFIG['REF_SEQ_DICT']

    # Check if file exists
    if not op.exists(fasta_filename):
        logging.info('Fasta filename cannot be found in path {}.'.format(fasta_filename))
    else:
        # Else, load trials
        with open(os.path.join(fasta_filename), 'r', newline=None) as f:
            logging.info('Loaded FASTA file...')

            # Fix varying length error
            records = list(SeqIO.parse(f, 'fasta'))
            # Make a copy, otherwise our generator is exhausted after calculating maxlen
            maxlen = max(len(record.seq) for record in records)
            # Pad sequences so that they all have the same length
            for record in records:
                if len(record.seq) != maxlen:
                    sequence = str(record.seq).ljust(maxlen, '-')
                    record.seq = Seq.Seq(sequence)
            assert all(len(record.seq) == maxlen for record in records)

            # Write to temporary file and do alignment
            padded_file = op.join(utils.data_path(MYPATH, 'raw'),'{}_padded.fasta'.format(f.name))

            with open(padded_file, 'w', newline=None) as pf:
                SeqIO.write(records, pf, 'fasta')

            # Load padded and perform alignment
            alignment = SeqIO.parse(padded_file, 'fasta')
            records = list(records)
            SEQ = [list(Seq.Seq(str(record.seq))) for record in records]
            # Shuffle order
            np.random.shuffle(SEQ)
            # Get alignment
            alignment = AlignIO.read(padded_file, 'fasta')
            # Find reference sequence given protein
            if protein:  # If protein given
                ref_seq = REF_SEQ_DICT[protein]
            else:
                protein, file_name, ref_seq = utils.infer_protein(filename=fasta_filename, logging=logging)
            # Check for mismatch
            if len(SEQ[1]) != len(ref_seq):
                logging.warning('Reference sequence dimension mismatch [len(ref_seq) != len(SEQ)] = [{len_ref} != {len_SEQ}]. Please check ref_seq_dict in src/data/make_dataset.py'.format(len_ref=len(ref_seq), len_SEQ=len(SEQ[0])))

            # Save and/or return file for modeling
            result = [SEQ, ref_seq]
            if pickle:
                final_alignment = op.join(utils.data_path(MYPATH, 'processed'), protein + '_seq.pkl')
                cPickle.dump([SEQ, ref_seq],open(final_alignment,"wb"))
                #hf = h5py.File(op.join(utils.data_path(MYPATH, 'processed'), protein + '_seq.h5'), 'w')
                #SEQ_encoded = np.char.encode(np.array([SEQ, ref_seq]), encoding='utf8')
                #hf.create_dataset(protein+'_seq', data=SEQ_encoded)
                #hf.close()
                #logging.info('[!] Data successfully saved in h5 format with dimensions [{len_fasta}, {len_SEQ}]'.format(len_fasta=len(SEQ), len_SEQ=len(SEQ[0])))
                logging.info('[!] Data successfully saved with dimensions [{len_fasta}, {len_SEQ}]'.format(len_fasta=len(SEQ), len_SEQ=len(SEQ[0])))
            return result

def load_pdb(pdb_filename, protein, logging=None):
    """
    Loads and processes PDB file

    Parameters
    ----------
    pdb_filename : str
        Filepath for PDB file.
    protein: str,
        Name of protein to collect data.
    logging : object
        Logging object

    Returns
    -------
    No returns, but saves inter residue distances in ../data/processed/{protein}_res_distances.pkl
    """
    CONFIG = utils.load_config()
    protein_dict = CONFIG['PROTEIN_DICT']

    if not protein:
        # If protein name not given, infer from pdb filename
        protein, file_name, ref_seq = utils.infer_protein(filename=pdb_filename, ext='pdb', logging=logging)
    if protein:
        # If protein name given, infer pdb filename
        file_name = protein_dict[protein]['pdb']

    # Parse the pdb file into a PDB.Structure object
    PDBparser = PDB.PDBParser(QUIET=True)  # disabled warnings
    struct = PDBparser.get_structure(file_name, pdb_filename)
    residues = list(struct.get_residues())

    # Calculate the distance between two positions
    res_start = protein_dict[protein]['start']
    res_end = protein_dict[protein]['end']
    Di_mat = []

    for p_i in range(res_start, res_end):
        for p_j in range(res_start, res_end):
            res_one = residues[int(p_i)]
            res_two = residues[int(p_j)]
            if res_one.id[0] == ' ' and res_two.id[0] ==' ':  # filters heteroatoms
                alpha_dist = res_one['CA'] - res_two['CA']
                Di_mat.append([p_i - res_start, p_j - res_start, alpha_dist])
    Di_mat_df=pd.DataFrame(Di_mat)
    distance_file = op.join(utils.data_path(MYPATH, 'processed'), protein + '_res_distances.pkl')
    cPickle.dump(Di_mat_df, open(distance_file, "wb"))
    logging.info('[!] Inter-residue distances saved')


def open_dataset(protein, fasta, pdb, logging):
    """
    Determines filenames given protein or fasta/pdb pair. Initiates data processing and saving.

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

    Returns
    -------
    Although nothing is returned, the functions load_fasta and load_pdb will save sequence, reference sequence, and inter residue distances to
    ..data/processed/
    """
    fasta_filename, pdb_filename = fetch_filename(protein, fasta, pdb, logging=logging)
    load_fasta(fasta_filename=fasta_filename, protein=protein, logging=logging, pickle=True)
    #load_pdb(pdb_filename=pdb_filename, protein=protein, logging=logging)


def main(protein, fasta='', pdb='', logging=None):
    """
    This function performs the data cleaning and processing necessary for
    modeling for the given fasta and pdb file.

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
        logging.info('Starting data acquisition and cleaning...')
        open_dataset(protein=protein, fasta=fasta, pdb=pdb, logging=logging)
        logging.info('Data loaded and processed successfully. Move on!')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers. Must provide EITHER protein or fasta/pdb combo.')
    parser.add_argument('--protein', type=str, help='Protein name.')
    parser.add_argument('--fasta', type=str, help='Fasta file name.', default='')
    parser.add_argument('--pdb', type=str, help='PDB file name.', default='')
    parser.add_argument('--logging', type=str, help='Logging option.', default=utils.logger())

    args = vars(parser.parse_args())
    main(**args)
