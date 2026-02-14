# RAPO Training Guide

This repo now includes a trainable RAPO extension:

- Model: `affectgpt_rapo`
- Auxiliary losses:
  - multi-label emotion supervision (`rapo_aux_loss_weight`)
  - confidence supervision for selective prediction (`rapo_conf_loss_weight`)

## 1) Run training

```bash
python train.py --cfg-path train_configs/ovmerd_rapo_train.yaml
```

## 2) Key implementation files

- `my_affectgpt/models/affectgpt_rapo.py`
- `my_affectgpt/models/rapo_label_utils.py`
- `my_affectgpt/datasets/datasets/base_dataset.py`
- `train_configs/ovmerd_rapo_train.yaml`

## 3) Behavior details

- Original LM loss is still active.
- For each batch, supervision is extracted from dataset samples with fallback priority:
  - `ovlabel` -> `onehot` -> `sentiment` -> `valence`.
- RAPO heads are only used when valid supervision labels are available.

## 4) Config knobs

- `model.rapo_aux_loss_weight`: weight of multi-label BCE objective.
- `model.rapo_conf_loss_weight`: weight of confidence regression objective.
- `model.rapo_vocab`: configurable label vocabulary.
- `model.rapo_vocab_path`: optional JSON vocabulary file.

## 5) Static checks

```bash
python -m compileall my_affectgpt
python -m py_compile my_affectgpt/models/affectgpt_rapo.py
python -m py_compile my_affectgpt/models/rapo_label_utils.py
python -m py_compile my_affectgpt/datasets/datasets/base_dataset.py
```
