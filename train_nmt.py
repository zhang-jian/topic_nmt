import numpy
import os

import numpy
import sys

from nmt import train

def main(job_id, params):
    print params
    f = 1000
    validerr = train(saveto=params['model'][0],
		     log=params['log'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=2000,
		     finish_after=320000,
                     maxlen=80,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=f,
                     dispFreq=f,
                     saveFreq=f,
                     sampleFreq=f,
                     datasets=['train.zh',
                               'train.en'],
                     valid_datasets=['train.zh',
                                     'train.en'],
                     dictionaries=['train.zh.pkl',
                                   'train.en.pkl'],
		     target_topic_dict=params['target_topic_dict'][0],
          	     target_topic_number=params['target_topic_number'][0],
		     source_topic_dict=params['source_topic_dict'][0],
          	     source_topic_number=params['source_topic_number'][0],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    saveto = "test"
    log = "log"
    source_topic_dict = "train.zh.t40"
    source_topic_number = 40
    target_topic_dict = "train.en.t10"
    target_topic_number = 10
    main(0, {
        'model': [saveto + '/model.npz'],
	'log':[log],
	'source_topic_dict':[source_topic_dict],
	'source_topic_number':[source_topic_number],
        'target_topic_dict':[target_topic_dict],
	'target_topic_number':[target_topic_number],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [16000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [True],
        'learning-rate': [0.0001],
        'reload': [True]})