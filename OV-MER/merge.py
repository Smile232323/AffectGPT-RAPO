## merge file into npz
import os
import glob
import numpy as np


# for model_root in glob.glob('output/*'):
#     for process_dir in glob.glob(model_root + '/*'):
#         save_path = process_dir + '.npz'

#         # => name2reason
#         name2reason = {}
#         for file_path in glob.glob(process_dir + '/*'):
#             reason = np.load(file_path).tolist()
#             name = os.path.basename(file_path)[:-4]
#             name2reason[name] = reason
        
#         np.savez_compressed(save_path, name2reason=name2reason)


from evaluation import *

## Clue merging: nosubtitle -> nosubtitle-addsub
if __name__ == "__main__":
    for modelname in ['SALMONN', 'Chat-UniVi', 'LLaMA-VID', 'mPLUG-Owl', 'Otter', 'Qwen-Audio', 
                      'VideoChat', 'VideoChat2', 'Video-ChatGPT','Video-LLaVA']:
        
        # 1. name2reason
        print (f'============ modelname: {modelname} ====================')
        reason_npz = f"output/results-ovmerd/{modelname}/results-nosubtitle.npz"  
        print (reason_npz)
        assert os.path.exists(reason_npz)
        name2reason = np.load(reason_npz, allow_pickle=True)['name2reason'].tolist()

        # name2subtitle
        dataset = func_read_datasetname(reason_npz)
        dataset_cls = get_dataset2cls(dataset)
        name2subtitle = dataset_cls.name2subtitle
        print (f'target sample number: {len(name2subtitle)}')

        # load model
        llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
        
        # main process
        save_npz = f"output/results-ovmerd/{modelname}/results-nosubtitle-addsub.npz"
        if not os.path.exists(save_npz):
            clue_merge_batchcalling(name2reason=name2reason, store_npz=save_npz, name2subtitle=name2subtitle,
                                    llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
            