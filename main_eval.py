import sys; sys.path.append("..")
from importlib import reload
import persist_to_disk as ptd
import os
ptd.config.set_project_path(os.path.abspath("../"))
import tqdm
import pandas as pd
import numpy as np
import re
import torch
import utils
import csv
import argparse

from _settings import GEN_PATHS

import matplotlib.pyplot as plt
# matplotlib inline

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

import pipeline.uq_bb as uq_bb

def get_args():
    parser = argparse.ArgumentParser(description='UQ Testing')
    parser.add_argument('--models', type=str, default='llama-13b', help='net type')
    parser.add_argument('--datas', type=str, default= 'coqa', help='dataset')
    parser.add_argument('--temperature', type=float, default='1.5')
    parser.add_argument('--eigv_threshold', type=float, default='0.9')
    parser.add_argument('--gpt_thr', type=float, default='0.7')
    parser.add_argument('--rougeL_thr', type=float, default='0.3')
    parser.add_argument('--n_components', type=int, default='32')
    
    
    
#     default_params = {'eigv_threshold': 0.9, 'temperature': 3., 'gpt_thr': 0.7, 'rougeL_thr':0.3 }
    
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args

def write_csv(filename, data):
    with open(filename, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)
        

def main_result(path, num_gens, summ_kwargs, device):
    
    obj = uq_bb.UQ_summ(path, eigv_threshold=args.eigv_threshold, temperature = args.temperature, gpt_thr = args.gpt_thr, rougeL_thr = args.rougeL_thr, n_components = args.n_components, clean=True, split='test', cal_size=1000, seed=1)
        
    summ_obj = obj.summ([
    'generations|numsets', 'lexical_sim',

            'generations|spectral_eigv_clip|disagreement_w',
            'generations|eccentricity|disagreement_w',
            'generations|degree|disagreement_w',

            'generations|spectral_eigv_clip|agreement_w',
            'generations|eccentricity|agreement_w',
            'generations|degree|agreement_w',


            'generations|spectral_eigv_clip|jaccard',
            'generations|eccentricity|jaccard',
            'generations|degree|jaccard',

#             'semanticEntropy|unnorm', 'self_prob',
    ], 

        acc_name='generations|gpt|acc',
        num_gens=num_gens, **summ_kwargs
    )
    
    u_ea_auarc = summ_obj.summ_overall('auarc')
    c_ia_auarc = sum(summ_obj.summ_individual('auarc', use_conf=True)) / num_gens
    c_ia_auroc = sum(summ_obj.summ_individual('auroc', use_conf=True)) / num_gens
    u_ea_auroc = sum(summ_obj.summ_individual('auroc', use_conf=False)) / num_gens
    
    return summ_obj, u_ea_auarc, c_ia_auarc, c_ia_auroc, u_ea_auroc



def main():
    
    result_file = "result.csv"
    write_csv(result_file, ['model','dataset','eigv_threshold','temperature','gpt_thr','rougeL_thr', 'n_components','u_ea_auarc', 'c_ia_auarc', 'c_ia_auroc', 'u_ea_auroc'])
    
    num_gens = 20
    summ_kwargs = {
        'u+ea': {'overall': True, 'use_conf': False},
        'u+ia': {'overall': False, 'use_conf': False},
        'c+ia': {'overall': False, 'use_conf': True},
    }['c+ia']
    
    models_all = ['llama-13b','opt-13b','gpt-3.5-turbo']
    data_all = ['coqa', 'nq_open', 'trivia']
    
    
    
    for models in models_all:
        for datas in data_all:
            args.models = models
            args.datas = datas
            path = GEN_PATHS[args.datas][args.models]  
            print('path', path)
            
            reload(uq_bb)
#             eigv_threshold=args.eigv_threshold, temperature = args.temperature, gpt_thr = args.gpt_thr, rougeL_thr = args.rougeL_thr,
            
            
            summ_obj, u_ea_auarc, c_ia_auarc, c_ia_auroc, u_ea_auroc = main_result(path, num_gens, summ_kwargs, device)
            
            write_csv(result_file, [str(args.models), str(args.datas), str(args.eigv_threshold), str(args.temperature), str(args.gpt_thr), str(args.rougeL_thr), str(args.n_components) ,str(u_ea_auarc),  str(c_ia_auarc), str(c_ia_auroc),str(u_ea_auroc)])


    
if __name__ == "__main__": 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args = get_args()
    print('temperature, threshold and pca_components',args.temperature, args.gpt_thr, args.n_components )
    main()