## Dataset paths
audio_root = {
    'emoprefer':          'dataset/audio',
    'emopreferreverse':   'dataset/audio',
    'emopreferv2':        'dataset/audio',
    'emopreferv2reverse': 'dataset/audio',
}

video_root = {
    'emoprefer':          'dataset/video',
    'emopreferreverse':   'dataset/video',
    'emopreferv2':        'dataset/video',
    'emopreferv2reverse': 'dataset/video',
}

label_csv = {
    'emoprefer':          'dataset/preference_emoprefer.csv',
    'emopreferreverse':   'dataset/preference_emoprefer_reverse.csv',
    'emopreferv2':        'dataset/preference_emopreferv2.csv',
    'emopreferv2reverse': 'dataset/preference_emopreferv2_reverse.csv',
}

label_withname_csv = {
    'emoprefer':          'dataset/preference_emoprefer_with_modelnames.csv',
    'emopreferv2':        'dataset/preference_emopreferv2_with_modelnames.csv',
}


## label mapping [reverse case only]
reverse_mapping = {
    'a1': 'a2',
    'a2': 'a1',
    'same': 'same',
}

## model -> model path
model2path = {
    'gemini_15_pro':        'models/gemini-1.5-pro-latest',
    'gemini_15_flash':      'models/gemini-1.5-flash-latest',
    'gemini_25_flash':      'models/gemini-2.5-flash-preview-05-20',
    'gemini_20_flash':      'models/gemini-2.0-flash',
    'gemini_20_flash_lite': 'models/gemini-2.0-flash-lite',

    'gpt_41':      'gpt-4.1',
    'gpt_41_mini': 'gpt-4.1-mini',
    'gpt_4o':      'gpt-4o',

    'qwen3_8b':  "models/Qwen3-8B",
    'qwen3_14b': "models/Qwen3-14B",
    'qwen25':    "models/Qwen2.5-7B-Instruct",

    'qwen25vl_3b':  "models/Qwen2.5-VL-3B-Instruct",
    'qwen25vl_7b':  "models/Qwen2.5-VL-7B-Instruct",
    'qwen25vl_32b': "models/Qwen2.5-VL-32B-Instruct",

    'qwen25omni_3b': "models/Qwen2.5-Omni-3B",
    'qwen25omni_7b': "models/Qwen2.5-Omni-7B",

    'qwen2audio': 'models/Qwen2-Audio-7B-Instruct',

    'llavanextvideo_7b':     'models/LLaVA-NeXT-Video-7B-hf',
    'llavanextvideo_7b_dpo': 'models/LLaVA-NeXT-Video-7B-DPO-hf',

    'vita_15':     'models/VITA-1.5',

    'vila_15_7b':  'models/VILA1.5-7b',
    'vila_15_13b': 'models/VILA1.5-13b',

    'pllava_7b':   'models/pllava-7b',
    'pllava_13b':  'models/pllava-13b',
}

## model -> default input type
model2input = {
    'gemini_15_pro':        'audiovideo',
    'gemini_15_flash':      'audiovideo',
    'gemini_25_flash':      'audiovideo',
    'gemini_20_flash':      'audiovideo',
    'gemini_20_flash_lite': 'audiovideo',

    'gpt_41':      'video',
    'gpt_41_mini': 'video',
    'gpt_4o':      'video',

    'videollava':      'video',
    'videochatgpt':    'video',
    'chatunivi_7b':    'video',
    'chatunivi_7b_15': 'video',
    'mplugowl':        'video',
    'otter':           'video',
    'llamavid':        'video',
    'videochat':       'video',
    'videochat2':      'video',

    'qwenaudio': 'audio',
    'salmonn':   'audio',

    'qwen25vl_3b':  'video',
    'qwen25vl_7b':  'video',
    'qwen25vl_32b': 'video',

    'qwen2audio': 'audio',

    'llavanextvideo_7b':     'video',
    'llavanextvideo_7b_dpo': 'video',

    'vita_15': 'video',

    'vila_15_7b': 'video',
    'vila_15_13b': 'video',

    'pllava_7b':  'video',
    'pllava_13b': 'video',
}
