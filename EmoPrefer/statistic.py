import os
import glob
import numpy as np
from utils.common import *
from toolkit.utils.functions import *
import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


########################################
## 全局变量
########################################
model_mapping = {
    'otter': 'Otter \\citep{li2023otter}',
    'videochat2': 'VideoChat2 \\citep{li2024mvbench}',
    'chatunivi_7b': 'Chat-UniVi \\citep{jin2024chat}',
    'videochatgpt': 'Video-ChatGPT \\citep{maaz2024video}',
    'mplugowl': 'mPLUG-Owl \\citep{ye2023mplugowl}',
    'videollava': 'Video-LLaVA \\citep{lin2024video}',
    'llamavid': 'LLaMA-VID \\citep{li2024llama}',
    'videochat': 'VideoChat \\citep{li2023videochat}',
    'pllava_7b': 'PLLAVA \\citep{xu2024pllava}',
    'vita_15': 'VITA-1.5 \\citep{fu2025vita}',
    'llavanextvideo_7b': 'LLaVA-Next-Video \\citep{li2024llava}',
    'gpt_4o': 'GPT-4o \\citep{openai24gpt4o}',
    'gemini_20_flash': 'Gemini-2.0-Flash \\citep{gemini20flash}',
    'qwen2audio': 'Qwen2-Audio \\citep{chu2024qwen2}',
    'gemini_25_flash': 'Gemini-2.5-Flash \\citep{gemini25flash}',
    'gpt_41': 'GPT-4.1 \\citep{achiam2023gpt}',
    'gemini_15_pro': 'Gemini-1.5-Pro \\citep{team2024gemini}',
    'qwen25vl_7b': 'Qwen2.5-VL \\citep{bai2025qwen2}',
    'gemini_15_flash': 'Gemini-1.5-Flash \\citep{team2024gemini}',
    'qwen25omni_7b': 'Qwen2.5-Omni \\citep{xu2025qwen2}',
    'affectgpt_mercaptionplus': 'AffectGPT \\citep{lian2025affectgpt}',
}

model_open  = ['otter','videochat2','chatunivi_7b','videochatgpt','mplugowl','videollava','llamavid','videochat',
               'pllava_7b','vita_15','llavanextvideo_7b','qwen2audio','qwen25vl_7b','qwen25omni_7b','affectgpt_mercaptionplus']

model_close = ['gpt_4o','gemini_20_flash','gemini_25_flash','gpt_41','gemini_15_pro','gemini_15_flash']


prompt_namemapping = {
    'normal': 'S1',
    'cot':    'S2',
    'cot2':   'S3',
    'cot3':   'S4',
}

##############################
## small function definition
##############################
# 1. 计算 multi-run 结果下的一致性
def func_consistency(save_npzs):
    whole_similarity = []
    for ii in range(len(save_npzs)):
        for jj in range(ii+1, len(save_npzs)):
            save_path_ii = save_npzs[ii]
            save_path_jj = save_npzs[jj]
            save_gt_ii = np.load(save_path_ii, allow_pickle=True)['gt_labels'].tolist()
            save_pd_ii = np.load(save_path_ii, allow_pickle=True)['pred_labels'].tolist()

            save_gt_jj = np.load(save_path_jj, allow_pickle=True)['gt_labels'].tolist()
            save_pd_jj = np.load(save_path_jj, allow_pickle=True)['pred_labels'].tolist()

            assert func_whether_two_list_are_same(save_gt_ii, save_gt_jj) is True
            same_precentage = func_two_list_same_precentage(save_pd_ii, save_pd_jj)
            whole_similarity.append(same_precentage)
    whole_similarity = np.array(whole_similarity) * 100
    return np.mean(whole_similarity), np.std(whole_similarity)

# 2. 计算 (forward, backward) 之间的一致性
def func_reverse_consistency(save_npzs, save_reverse_npzs):
    assert len(save_npzs) == len(save_reverse_npzs)

    whole_similarity = []
    for ii in range(len(save_npzs)):
        save_path_ii = save_npzs[ii]
        save_path_jj = save_reverse_npzs[ii]
        save_gt_ii = np.load(save_path_ii, allow_pickle=True)['gt_labels'].tolist()
        save_pd_ii = np.load(save_path_ii, allow_pickle=True)['pred_labels'].tolist()

        save_gt_jj = np.load(save_path_jj, allow_pickle=True)['gt_labels'].tolist()
        save_pd_jj = np.load(save_path_jj, allow_pickle=True)['pred_labels'].tolist()
        save_gt_jj = [config.reverse_mapping[item] for item in save_gt_jj]
        save_pd_jj = [config.reverse_mapping[item] for item in save_pd_jj]

        assert func_whether_two_list_are_same(save_gt_ii, save_gt_jj) is True
        same_precentage = func_two_list_same_precentage(save_pd_ii, save_pd_jj)
        whole_similarity.append(same_precentage)
    whole_similarity = np.array(whole_similarity) * 100
    return np.mean(whole_similarity), np.std(whole_similarity)

# 3. 获取 normal / reverse 进行 majority voting 后的结果
def func_normal_reverse_voting(save_npzs, save_reverse_npzs):
    whole_preds, whole_gt = [], []
    for npz in save_npzs:
        pred_labels = np.load(npz, allow_pickle=True)['pred_labels'].tolist()
        gt_labels   = np.load(npz, allow_pickle=True)['gt_labels'].tolist()
        whole_preds.append(pred_labels)
        whole_gt.append(gt_labels)
    
    for npz in save_reverse_npzs:
        pred_labels = np.load(npz, allow_pickle=True)['pred_labels'].tolist()
        gt_labels   = np.load(npz, allow_pickle=True)['gt_labels'].tolist()
        pred_labels = [config.reverse_mapping[item] for item in pred_labels]
        gt_labels   = [config.reverse_mapping[item] for item in gt_labels]
        whole_preds.append(pred_labels)
        whole_gt.append(gt_labels)
    assert func_whether_two_list_are_same(whole_gt[0], whole_gt[-1])
    voted_preds = func_majority_vote(whole_preds)
    return whole_gt[0], voted_preds

# 4. 将 list 转成 mapping 的形式
## names_number=3, keys_number=2 => 让 result_list 中，前三个是 name，后两个是 key
def func_convert_to_mapping(result_list, names_number=3, keys_number=2):
    nested_dict = {}
    
    def list_to_nested_dict(names, keys):
        current_level = nested_dict
        for item in names[:-1]:
            if item not in current_level:
                current_level[item] = {}
            current_level = current_level[item]
        current_level[names[-1]] = keys
        return nested_dict

    for item in result_list:
        assert len(item) == names_number + keys_number
        names = item[:names_number]
        keys  = item[names_number:]
        list_to_nested_dict(names, keys)
    
    return nested_dict

# 5. 从 map 中检索相应的值
def func_retrival(key2value, key_idx_lists):
    values = []
    for (key, idx) in key_idx_lists:
        values.append(key2value[key][idx])
    return values

# 6. 格式转换
def func_convert_format_meanstd(mean_std_list):
    return f'{mean_std_list[0]:.2f}$\\pm${mean_std_list[1]:.2f}'

def func_convert_format_meanonly(mean_std_list):
    return f'{mean_std_list[0]:.2f}'


##############################
## 计算所有统计结果，用于后面使用
##############################
def main_statistic(basefile='xxx'):
    
    whole_store = {}

    for (model, input_type) in [
        # ('qwen25vl_7b', 'video'),
        ('qwen25omni_7b', 'audiovideo'),
        # ('gemini_15_pro', 'audiovideo'),
        # ('gemini_15_flash', 'audiovideo'),
        # ('gemini_25_flash', 'audiovideo'),
        # ('gemini_20_flash', 'audiovideo'),
        # ('gpt_4o', 'video'),
        # ('gpt_41', 'video'),
        # ('videochatgpt', 'video'), 
        # ('videollava', 'video'),
        # ('qwen2audio', 'audio'),
        # ('llamavid', 'video'),
        # ('chatunivi_7b', 'video'),
        # ('mplugowl', 'video'),
        # ('otter', 'video'),
        # ('videochat', 'video'),
        # ('videochat2', 'video'),
        # ('llavanextvideo_7b', 'video'),
        # ('vita_15', 'video'),
        # ('pllava_7b', 'video'),
        ]:
        for prompt in ['normal', 'cot', 'cot2', 'cot3']:
            # print (f'====== process on {model} {prompt} ======')
            # per_store: 存储 (model, input_type, prompt) 下的所有结果
            per_store = {}

            if prompt == 'normal':
                save_npzs         = glob.glob(f'output-matching/{model}-{input_type}-normal-{basefile}-round*.npz')
                save_reverse_npzs = glob.glob(f'output-matching/{model}-{input_type}-normal-{basefile}reverse-round*.npz')
            elif prompt == 'cot':
                save_npzs         = glob.glob(f'output-matching/{model}-{input_type}-cot-answer-{basefile}-round*.npz')
                save_reverse_npzs = glob.glob(f'output-matching/{model}-{input_type}-cot-answer-{basefile}reverse-round*.npz')
            elif prompt == 'cot2':
                save_npzs         = glob.glob(f'output-matching/{model}-{input_type}-cot-answer2-{basefile}-qwen25-round*.npz')
                save_reverse_npzs = glob.glob(f'output-matching/{model}-{input_type}-cot-answer2-{basefile}reverse-qwen25-round*.npz')
            elif prompt == 'cot3':
                save_npzs         = glob.glob(f'output-matching/{model}-{input_type}-cot-answer3-{basefile}-qwen25-round*.npz')
                save_reverse_npzs = glob.glob(f'output-matching/{model}-{input_type}-cot-answer3-{basefile}reverse-qwen25-round*.npz')

            ####################################################################################
            ## debug: 测试临时跑完的文件的结果
            # print (len(save_npzs), len(save_reverse_npzs))
            ####################################################################################

            ## 计算 save_npz 下的结果
            if len(save_npzs) != 0:
                whole_metrics = []
                for save_npz in sorted(save_npzs):
                    waf_twoclass, acc_twoclass = func_preference_metric(save_npz, metric='twoclass')
                    waf_thrclass, acc_thrclass = func_preference_metric(save_npz, metric='threeclass')
                    whole_metrics.append([waf_twoclass, acc_twoclass, waf_thrclass, acc_thrclass])
                whole_metrics = np.array(whole_metrics) * 100
                avg_score1, avg_score2, avg_score3, avg_score4 = np.mean(whole_metrics, axis=0)
                std_score1, std_score2, std_score3, std_score4 = np.std(whole_metrics,  axis=0)
                per_store['two-class waf'] = [avg_score1, std_score1]
                per_store['two-class acc'] = [avg_score2, std_score2]
                per_store['thr-class waf'] = [avg_score3, std_score3]
                per_store['thr-class acc'] = [avg_score4, std_score4]
            
            # 展示正常与翻转情况下，结果的差异性，从而反映模型预测质量
            if len(save_npzs) != 0 and len(save_reverse_npzs) != 0 and len(save_npzs) == len(save_reverse_npzs):
                reverse_mean, reverse_std = func_reverse_consistency(save_npzs, save_reverse_npzs)
                per_store['reverse consistency'] = [reverse_mean, reverse_std]

            # 将 per_store 存储在 whole_store 中
            if len(per_store) != 0:
                if model not in whole_store: whole_store[model] = {}
                if prompt not in whole_store[model]: whole_store[model][prompt] = []
                whole_store[model][prompt] = per_store
    return whole_store

# 获取所有情况下的统计结果
# main_statistic(basefile='emoprefer')
# main_statistic(basefile='emopreferv2')


##############################################################################################
## 论文结果展示2：每种模型只展示一种prompt下的影响，选择 avg_score1 最高的部分结果
##############################################################################################
def main_present_results(basefile='xxx'):

    store_results = main_statistic(basefile)

    ## 结果展示部分
    show_models, show_metrics, show_results = [], [], []
    for model in store_results:

        # 1. model -> bestprompt
        prompts, two_class_waf_means = [], []
        for prompt in store_results[model]:
            prompts.append(prompt)
            two_class_waf_means.append(store_results[model][prompt]['two-class waf'][0])
        maxindex = np.argmax(two_class_waf_means)
        bestprompt = prompts[maxindex]

        # 2. (model, bestprompt) -> results
        two_class_waf = func_convert_format_meanstd(store_results[model][bestprompt]['two-class waf'])
        two_class_acc = func_convert_format_meanstd(store_results[model][bestprompt]['two-class acc'])
        thr_class_waf = func_convert_format_meanstd(store_results[model][bestprompt]['thr-class waf'])
        thr_class_acc = func_convert_format_meanstd(store_results[model][bestprompt]['thr-class acc'])
        reverse_cons  = func_convert_format_meanstd(store_results[model][bestprompt]['reverse consistency'])
        show_result = [model_mapping[model], prompt_namemapping[bestprompt], two_class_waf, two_class_acc, thr_class_waf, thr_class_acc, reverse_cons]
        show_result = "& ".join(show_result) + '\\\\'

        # 3. store: 将模型按照结果排序
        show_models.append(model)
        show_metrics.append(store_results[model][bestprompt]['two-class waf'][0])
        show_results.append(show_result)
    
    # sorted and show
    indexes = np.argsort(np.array(show_metrics))
    # 1. 将 open model 从小到大输出
    for index in indexes:
        if show_models[index] in model_open:
            print (show_results[index])
    
    # 2. 将 close model 从小到大输出
    print ('===================================')
    for index in indexes:
        if show_models[index] in model_close:
            print (show_results[index])
    
    return show_results


print ('EmoPrefer-Data + Qwen2.5-Omni: ')
main_present_results(basefile='emoprefer')
'''
Qwen2.5-Omni \citep{xu2025qwen2}& S2& 67.21$\pm$0.00& 67.32$\pm$0.00& 65.29$\pm$0.00& 66.03$\pm$0.00& 79.09$\pm$0.00\\
'''

print ('EmoPrefer-Data-V2 + Qwen2.5-Omni: ')
main_present_results(basefile='emopreferv2')
'''
Qwen2.5-Omni \citep{xu2025qwen2}& S1& 64.96$\pm$0.00& 66.03$\pm$0.00& 43.93$\pm$0.00& 51.19$\pm$0.00& 71.47$\pm$0.00\\
'''