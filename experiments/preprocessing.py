import sys
import os
import pylidc as pl
import numpy as np
import better_exceptions
from utils import get_patches, save_patch
import sklearn.model_selection
import pickle
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()
config.read('./configs.ini')

def scan_index_split(number_scans, rs=42):
    train, test = sklearn.model_selection.train_test_split(list(range(number_scans)), train_size=0.8, random_state=rs)
    train, valid = sklearn.model_selection.train_test_split(train, train_size=0.8, random_state=rs)
    return train, valid, test

def generate_patches():
    data_dir            = config['Lung']['DataDirectory']
    scans               = pl.query(pl.Scan).filter()
    train, valid, test  = scan_index_split(scans.count())
    if os.path.isfile(os.path.join(data_dir, 'infos.pl')):
        infos = pickle.load(open(os.path.join(data_dir, 'infos.pl'), 'rb'))
    else:
        infos = dict()
        infos['start_scan_index']   = 0
        infos['train_size']        = 0
        infos['valid_size']        = 0
        infos['test_size']         = 0

    for index_scan, scan in enumerate(tqdm(scans, desc='Patch Gen', dynamic_ncols=True, total=scans.count())):
        if index_scan < infos['start_scan_index']:
            continue

        if index_scan in test:      which_set = 'test'
        elif index_scan in valid:   which_set = 'valid'
        else:                       which_set = 'train'

        try:
            X, y = get_patches(scan)
        except KeyboardInterrupt: # ctrl-c
            break
        except:
            infos[index_scan] = {
                'start': -1,
                'size': -1
            }
        else:
            infos[index_scan] = {
                'start': infos[which_set+'_size'],
                'size': X.shape[0]
            }

            # save patch
            for xi, yi in zip(X, y):
                save_patch(xi, yi, infos[which_set+'_size'], which_set)
                infos[which_set+'_size'] += 1

        infos['start_scan_index'] = index_scan+1
        pickle.dump(infos, open(os.path.join(data_dir, 'infos.pl'), 'wb'))

if __name__ == '__main__':
    generate_patches()
