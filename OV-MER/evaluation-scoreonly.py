import os
import re
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import datetime

import config
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
import numpy as np
from toolkit.utils.read_files import *
from toolkit.utils.qwen import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.wheel import func_get_name2reason
from my_affectgpt.datasets.builders.image_text_pair_builder import * # 加载所有dataset cls
from my_affectgpt.evaluation.ew_metric import *
from my_affectgpt.evaluation.wheel import *

def search_for_result_root(input_dir, inter_print=True):
    candidates = glob.glob(input_dir + '*')
    root_candidates = [root for root in candidates if os.path.isdir(root)]
    if len(root_candidates) == 0:
        if inter_print: print ('No file exists!')
        return ''
    
    # 找到 files 最多的 root
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        store_path = []
        for path in os.listdir(root):
            if path.startswith('checkpoint_') and path.find('-') == -1:
                store_path.append(path)
        count = len(store_path)
        if inter_print: print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root

    if inter_print: print ('================================================')
    if inter_print: print ('Targetroot: ', targetroot)
    if inter_print: print ('Saved result files ', maxcount)
    # report last file info
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    if inter_print: print("Last result file creation time:", datetime.datetime.fromtimestamp(creation_time))
    if inter_print: print ('================================================')
    return targetroot


def func_read_datasetname(input_dir):
    # print (input_dir)
    supprot_datasets = list(config.DATA_DIR.keys())
    assert input_dir.find('/results-') != -1
    dataset = input_dir.split('/results-')[1].split('/')[0]
    for supprot_item in supprot_datasets:
        if supprot_item.lower() == dataset.lower():
            return supprot_item
    ValueError(f'cannot find suitable dataset for {input_dir}')


def get_dataset2cls(dataset):
    if dataset == 'MER2023':     return MER2023_Dataset()
    if dataset == 'MER2024':     return MER2024_Dataset()
    if dataset == 'MELD':        return MELD_Dataset()
    if dataset == 'IEMOCAPFour': return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI':     return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI':    return CMUMOSEI_Dataset()
    if dataset == 'SIMS':        return SIMS_Dataset()
    if dataset == 'SIMSv2':      return SIMSv2_Dataset()
    if dataset == 'OVMERD':      return OVMERD_Dataset()
    if dataset == 'MER2025OV':        return MER2025OV_Dataset()
    print ('dataset cls not provided!')
    return None


def get_discrete_or_dimension_flag(dataset):
    if dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour']:
        return 'discrete'
    elif dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']:
        return 'dimension'
    elif dataset in ['MER2025OV', 'OVMERD']:
        return 'ovlabel'
    else:
        ValueError('unsupported dataset input')


def get_emo2idx_idx2emo(dataset_cls):
    emo2idx, idx2emo = {}, {}

    if hasattr(dataset_cls, 'get_emo2idx_idx2emo'): 
        emo2idx, idx2emo = dataset_cls.get_emo2idx_idx2emo()
        # post process [不同数据集的标签表示有些许差异，进行统一化处理]
        if 'happy' in emo2idx: emo2idx['joy']   = emo2idx['happy']
        if 'anger' in emo2idx: emo2idx['angry'] = emo2idx['anger']
        if 'sad'   in emo2idx: emo2idx['sadness'] = emo2idx['sad']
        if 'joy'   in emo2idx: emo2idx['happy'] = emo2idx['joy']
        if 'angry' in emo2idx: emo2idx['anger'] = emo2idx['angry']
    return emo2idx, idx2emo

def func_read_batch_calling_model(modelname):
    model_path = config.PATH_TO_LLM[modelname]
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    return llm, tokenizer, sampling_params


def calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print=True):
    
    # epoch_root=(name2reason) => openset
    openset_npz = epoch_root[:-4]+'-openset.npz'
    if not os.path.exists(openset_npz):
        extract_openset_batchcalling(reason_npz=epoch_root, store_npz=openset_npz,
                                     llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
        
    # 计算 EW-based metrics
    name2pred = {}
    filenames = np.load(openset_npz, allow_pickle=True)['filenames']
    fileitems = np.load(openset_npz, allow_pickle=True)['fileitems']
    for (name, item) in zip(filenames, fileitems):
        name2pred[name] = item
    whole_fscores = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, inter_print=inter_print)
    return whole_fscores


def main_zeroshot_scores(input_dir, debug=False, test_epochs='', inter_print=True):

    # ## 如果 input_dir 不存在的话，那么需要去检索最匹配的路径
    if not os.path.exists(input_dir):
        input_dir = search_for_result_root(input_dir, inter_print)
    if inter_print: print (f'process root: {input_dir}')

    # read dataset infos
    dataset = func_read_datasetname(input_dir)
    disordim_flag = get_discrete_or_dimension_flag(dataset)
    if inter_print: print (f'process dataset: {dataset} => {disordim_flag}')
    dataset_cls = get_dataset2cls(dataset)
    name2gt = dataset_cls.get_test_name2gt()
    if inter_print: print (f'target sample number: {len(name2gt)}')

    # load model
    llm, tokenizer, sampling_params = None, None, None
    if debug == False:
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    
    # main process
    for epoch_root in sorted(glob.glob(input_dir + '/*.npz')):

        if epoch_root.find('openset') != -1 or epoch_root.find('sentiment') != -1:
            continue

        # =============== process for {epoch_root} ===============
        epochname = os.path.basename(epoch_root)
        if inter_print: print (epochname)
        # 0. 判断 epoch 是不是在 test_epochs 内，否是就跳过这部分
        if test_epochs != '':
            run_epochs = [int(item) for item in test_epochs.split(',')]
            cur_epoch = int(epochname.split('_')[1])
            if cur_epoch not in run_epochs: continue

        assert disordim_flag == 'ovlabel'
        whole_fscore = calculate_ov_zeroshot(epoch_root, name2gt, llm, tokenizer, sampling_params, inter_print)
        whole_fscore = ['%.2f'%(item*100) for item in whole_fscore]
        print (os.path.basename(epoch_root), whole_fscore)
        

if __name__ == "__main__":

    ## step1：测试新模型下的结果
    # for dataset in ['ovmerd']:
    #     for modelname in ['Chat-UniVi', 'LLaMA-VID',  'mPLUG-Owl',  'Otter',  'Qwen-Audio', 
    #                        'SALMONN' , 'VideoChat'  ,'VideoChat2' , 'Video-ChatGPT'  ,'Video-LLaVA']:
    #         main_zeroshot_scores(f"output/results-{dataset}/{modelname}")


    # ## step2: 结果汇总展示
    for modelname in ['Chat-UniVi', 'LLaMA-VID', 'mPLUG-Owl', 'Otter', 'Qwen-Audio', 
                      'SALMONN', 'VideoChat', 'VideoChat2', 'Video-ChatGPT','Video-LLaVA']:
        print (f'============ modelname: {modelname} ====================')
        process_root = f"output/results-ovmerd/{modelname}"
        main_zeroshot_scores(process_root, debug=True, test_epochs='', inter_print=False)
