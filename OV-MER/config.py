# *_*coding:utf-8 *_*
import os

## 所有涉及 transformers 的模型存储路径
AFFECTGPT_ROOT = './'
EMOTION_WHEEL_ROOT = './emotion_wheel'
RESULT_ROOT = os.path.join(AFFECTGPT_ROOT, 'output/results')


#######################
## 所有模型的存储路径
#######################
PATH_TO_MLLM = {
    ## For Qwen-Audio
    'qwen-audio-chat':            '../models/qwen-audio-chat',
    ## For SALMONN
    'salmonn_7b':                 '../models/salmonn_7b.pth',
    'vicuna-7b-v1.5':             '../models/vicuna-7b-v1.5',
    'BEATs':                      '../models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt', 
    'whisper-large-v2':           '../models/whisper-large-v2', 
    ## For Video-ChatGPT
    'video_chatgpt-7B':           '../models/video_chatgpt-7B.bin',
    'LLaVA-7B-Lightening-v1-1':   '../models/LLaVA-7B-Lightening-v1-1',
    'clip-vit-large-patch14':     '../models/clip-vit-large-patch14',
    ## For Video-LLaMA
    'llama-2-7b-chat-hf':         '../models/llama-2-7b-chat-hf',
    'imagebind_huge':             '../models/imagebind_huge.pth',
    'video_llama_vl':             '../models/VL_LLaMA_2_7B_Finetuned.pth',
    'video_llama_al':             '../models/AL_LLaMA_2_7B_Finetuned.pth',
    'blip2_pretrained_flant5xxl': '../models/blip2_pretrained_flant5xxl.pth',
    'bert-base-uncased':          '../models/bert-base-uncased',
    'eva_vit_g':                  '../models/eva_vit_g.pth',
    ## For Chat-UniVi
    'Chat-UniVi':                 '../models/Chat-UniVi',
    ## For LLaMA-VID
    'llama-vid':                  '../models/llama-vid-7b-full-224-video-fps-1',
    ## For mPLUG-Owl
    'mplug-owl':                  '../models/mplug-owl-llama-7b-video',
    ## For Otter
    'otter':                      '../models/OTTER-Video-LLaMA7B-DenseCaption',
    ## For VideoChat
    'vicuna-7b-v0':               '../models/vicuna-7b-v0',
    'videochat_7b':               '../models/videochat_7b.pth',
    ## For VideoChat2
    'umt_l16_qformer':            '../models/umt_l16_qformer.pth',
    'videochat2_7b_stage2':       '../models/videochat2_7b_stage2.pth',
    'videochat2_7b_stage3':       '../models/videochat2_7b_stage3.pth',
    ## For Video-LLaVA
    'Video-LLaVA':                '../models/Video-LLaVA-7B',
}

PATH_TO_LLM = {
    'Qwen25': 'models/Qwen2.5-7B-Instruct',
}

#######################
## 所有数据集的存储路径
#######################
DATA_DIR = {
    'MER2025OV':      'xxx/dataset/mer2025-dataset',
    'MERCaptionPlus': 'xxx/dataset/mer2025-dataset',
    'OVMERD':         'xxx/dataset/mer2025-dataset',
    'MER2023':        'xxx/dataset/mer2023-dataset-process',
    'MER2024':        'xxx/dataset/mer2024-dataset-process',
    'IEMOCAPFour':    'xxx/dataset/iemocap-process',
    'CMUMOSI':        'xxx/dataset/cmumosi-process',
    'CMUMOSEI':       'xxx/dataset/cmumosei-process', 
    'SIMS':           'xxx/dataset/sims-process',
    'SIMSv2':         'xxx/dataset/simsv2-process',
    'MELD':           'xxx/dataset/meld-process', 
}
PATH_TO_RAW_AUDIO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'audio'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'audio'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'audio'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'audio'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subaudio'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'audio'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subaudio'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'audio'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'audio'),
}
PATH_TO_RAW_VIDEO = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'video'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'video'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'video'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'video'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'subvideo-tgt'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subvideo'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subvideo_new'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'video'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'subvideo'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'video_new'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'video'),

}
PATH_TO_RAW_FACE = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'openface_face'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'openface_face'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'openface_face'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'openface_face'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'openface_face'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'openface_face'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'openface_face'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'subtitle_chieng.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'subtitle_chieng.csv'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'subtitle_chieng.csv'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'transcription-engchi-polish.csv'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'transcription-engchi-polish.csv'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription-engchi-polish.csv'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription-engchi-polish.csv'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'transcription-engchi-polish.csv'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'transcription-engchi-polish.csv'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'transcription-engchi-polish.csv'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'transcription_merge.csv'),
}
PATH_TO_LABEL = {
    'MER2025OV':  os.path.join(DATA_DIR['MER2025OV'], 'track2_test.csv'),
    'MERCaptionPlus':  os.path.join(DATA_DIR['MERCaptionPlus'], 'xxx'),
    'OVMERD':  os.path.join(DATA_DIR['OVMERD'], 'xxx'),
    'MER2023': os.path.join(DATA_DIR['MER2023'], 'label-6way.npz'),
    'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'label_4way.npz'),
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'label.npz'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'label.npz'),
    'SIMS': os.path.join(DATA_DIR['SIMS'], 'label.npz'),
    'MELD': os.path.join(DATA_DIR['MELD'], 'label.npz'),
    'SIMSv2': os.path.join(DATA_DIR['SIMSv2'], 'label.npz'),
    'MER2024': os.path.join(DATA_DIR['MER2024'], 'label-6way.npz'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
