import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_affectgpt.common.registry import registry
from my_affectgpt.models.affectgpt import AffectGPT
from my_affectgpt.models.rapo_label_utils import (
    DEFAULT_ALIAS,
    build_vocab,
    label_count_to_confidence,
    labels_to_multihot,
)


@registry.register_model("affectgpt_rapo")
class AffectGPTRapo(AffectGPT):
    """
    RAPO extension:
    - keep original LM objective
    - add multi-label auxiliary loss over emotion vocabulary
    - add confidence regression head for selective prediction
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/affectgpt.yaml",
    }

    def __init__(
        self,
        visual_encoder_name,
        acoustic_encoder_name,
        llama_model_name,
        frozen_video_proj,
        frozen_video_Qformer,
        frozen_audio_Qformer,
        frozen_audio_proj,
        frozen_llm,
        lora_r,
        num_video_query_token,
        num_audio_query_token,
        num_multi_query_token,
        num_image_query_token,
        frozen_multi_Qformer,
        frozen_multi_llama_proj,
        multi_fusion_type,
        video_fusion_type,
        audio_fusion_type,
        image_fusion_type,
        rapo_aux_loss_weight=0.2,
        rapo_conf_loss_weight=0.1,
        rapo_vocab=None,
        rapo_vocab_path="",
        rapo_alias_map=None,
    ):
        super().__init__(
            visual_encoder_name=visual_encoder_name,
            acoustic_encoder_name=acoustic_encoder_name,
            llama_model_name=llama_model_name,
            frozen_video_proj=frozen_video_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            frozen_audio_proj=frozen_audio_proj,
            frozen_llm=frozen_llm,
            lora_r=lora_r,
            num_video_query_token=num_video_query_token,
            num_audio_query_token=num_audio_query_token,
            num_multi_query_token=num_multi_query_token,
            num_image_query_token=num_image_query_token,
            frozen_multi_Qformer=frozen_multi_Qformer,
            frozen_multi_llama_proj=frozen_multi_llama_proj,
            multi_fusion_type=multi_fusion_type,
            video_fusion_type=video_fusion_type,
            audio_fusion_type=audio_fusion_type,
            image_fusion_type=image_fusion_type,
        )

        self.rapo_aux_loss_weight = float(rapo_aux_loss_weight)
        self.rapo_conf_loss_weight = float(rapo_conf_loss_weight)

        vocab = build_vocab(cfg_vocab=rapo_vocab, vocab_path=rapo_vocab_path)
        self.rapo_vocab: List[str] = vocab
        self.rapo_vocab_to_idx: Dict[str, int] = {x: i for i, x in enumerate(vocab)}

        alias_map = dict(DEFAULT_ALIAS)
        if isinstance(rapo_alias_map, dict):
            for k, v in rapo_alias_map.items():
                kk = str(k).strip().lower()
                vv = str(v).strip().lower()
                if kk and vv:
                    alias_map[kk] = vv
        self.rapo_alias_map = alias_map

        hidden_size = int(self.llama_model.config.hidden_size)
        self.rapo_label_head = nn.Linear(hidden_size, len(self.rapo_vocab))
        self.rapo_conf_head = nn.Linear(hidden_size, 1)

        print(
            f"RAPO enabled: vocab={len(self.rapo_vocab)} "
            f"aux_w={self.rapo_aux_loss_weight} conf_w={self.rapo_conf_loss_weight}"
        )

    def _build_modal_pooled_feature(
        self,
        frame_llms=None,
        face_llms=None,
        audio_llms=None,
        image_llms=None,
        multi_llms=None,
    ):
        modal_feats = []
        for llm_feats in [frame_llms, face_llms, audio_llms, image_llms, multi_llms]:
            if llm_feats is None:
                continue
            modal_feats.append(llm_feats.mean(dim=1))
        if not modal_feats:
            return None
        stacked = torch.stack(modal_feats, dim=1)
        pooled = stacked.mean(dim=1)
        return pooled

    def _compute_rapo_losses(
        self,
        pooled_feature,
        supervision_texts,
        device,
        supervision_available=None,
    ):
        zero = torch.tensor(0.0, device=device)
        if pooled_feature is None:
            return zero, zero
        if supervision_texts is None:
            return zero, zero
        if len(supervision_texts) == 0:
            return zero, zero

        targets, valid_mask, label_counts = labels_to_multihot(
            supervision_texts,
            vocab_to_idx=self.rapo_vocab_to_idx,
            alias_map=self.rapo_alias_map,
        )
        valid = torch.tensor(valid_mask, device=device, dtype=torch.bool)

        if supervision_available is not None:
            if torch.is_tensor(supervision_available):
                available = supervision_available.to(device=device).bool()
            else:
                available = torch.tensor(supervision_available, device=device, dtype=torch.bool)
            if available.numel() != valid.numel():
                raise ValueError(
                    f"supervision_available size mismatch: "
                    f"{available.numel()} vs {valid.numel()}"
                )
            valid = valid & available

        if valid.sum().item() == 0:
            return zero, zero

        target_tensor = torch.tensor(
            targets,
            device=device,
            dtype=torch.float32,
        )
        # Compute auxiliary losses in fp32 for better numerical stability.
        logits = self.rapo_label_head(pooled_feature).float()
        aux_loss = F.binary_cross_entropy_with_logits(
            logits[valid], target_tensor[valid], reduction="mean"
        )

        conf_target = [label_count_to_confidence(x) for x in label_counts]
        conf_target = torch.tensor(conf_target, device=device, dtype=torch.float32)
        conf_pred = torch.sigmoid(self.rapo_conf_head(pooled_feature).float()).squeeze(-1)
        conf_loss = F.smooth_l1_loss(conf_pred[valid], conf_target[valid], reduction="mean")
        return aux_loss, conf_loss

    def forward(self, samples):
        self.face_or_frame = samples["face_or_frame"]
        frame_llms, face_llms, audio_llms, image_llms, multi_llms = None, None, None, None, None

        if "frames" in samples:
            frame_hiddens, frame_llms = self.encode_video_merge(samples["frames"], samples["raw_frames"])
        if "faces" in samples:
            face_hiddens, face_llms = self.encode_video_merge(samples["faces"], samples["raw_faces"])
        if "audios" in samples:
            audio_hiddens, audio_llms = self.encode_audio_merge(samples["audios"], samples["raw_audios"])
        if "images" in samples:
            image_hiddens, image_llms = self.encode_image_merge(samples["images"], samples["raw_images"])

        if (samples["input_ids"][0] == self.MULTI_PATCH_TOKEN_ID).sum() != 0:
            if self.face_or_frame.startswith("multiface"):
                multi_hiddens, multi_llms = self.encode_multi_merge(face_hiddens, audio_hiddens)
            if self.face_or_frame.startswith("multiframe"):
                multi_hiddens, multi_llms = self.encode_multi_merge(frame_hiddens, audio_hiddens)

        pooled_feature = self._build_modal_pooled_feature(
            frame_llms=frame_llms,
            face_llms=face_llms,
            audio_llms=audio_llms,
            image_llms=image_llms,
            multi_llms=multi_llms,
        )

        input_ids = samples["input_ids"]
        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == self.FRAME_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.FACE_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.MULTI_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.IMAGE_PATCH_TOKEN_ID] = 0
        temp_input_embedding = self.llama_model.model.model.embed_tokens(temp_input_ids)

        cur_idx = 0
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
            for (patch_token_id, query_token_number, embeds) in [
                (self.FRAME_PATCH_TOKEN_ID, self.num_video_query_token, frame_llms),
                (self.FACE_PATCH_TOKEN_ID, self.num_video_query_token, face_llms),
                (self.AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, audio_llms),
                (self.MULTI_PATCH_TOKEN_ID, self.num_multi_query_token, multi_llms),
                (self.IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, image_llms),
            ]:
                if (cur_input_ids == patch_token_id).sum() != 0:
                    assert embeds is not None, "Some input info is missing."
                    cur_features = embeds[cur_idx]
                    if (cur_input_ids == patch_token_id).sum() != query_token_number:
                        raise ValueError(
                            "The number of patch tokens should be the same as the number of patches."
                        )
                    masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                    mask_index_start = masked_indices[0]
                    expected = torch.arange(
                        mask_index_start,
                        mask_index_start + query_token_number,
                        device=masked_indices.device,
                        dtype=masked_indices.dtype,
                    )
                    if (masked_indices != expected).any():
                        raise ValueError("Patch tokens should be consecutive.")
                    cur_input_embeds = torch.cat(
                        (
                            cur_input_embeds[:mask_index_start],
                            cur_features,
                            cur_input_embeds[mask_index_start + query_token_number :],
                        ),
                        dim=0,
                    )
            new_input_embeds.append(cur_input_embeds)
            cur_idx += 1
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        targets = samples["labels"]
        attention_mask = samples["attention_masks"]
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        lm_loss = outputs.loss

        aux_loss, conf_loss = self._compute_rapo_losses(
            pooled_feature=pooled_feature,
            supervision_texts=samples.get("supervision_texts", None),
            device=lm_loss.device,
            supervision_available=samples.get("supervision_available", None),
        )
        loss = (
            lm_loss
            + self.rapo_aux_loss_weight * aux_loss
            + self.rapo_conf_loss_weight * conf_loss
        )
        return {
            "loss": loss,
            "loss_lm": lm_loss.detach(),
            "loss_aux": aux_loss.detach(),
            "loss_conf": conf_loss.detach(),
        }

    @classmethod
    def from_config(cls, cfg):
        visual_encoder_name = cfg.get("visual_encoder", "xxx")
        acoustic_encoder_name = cfg.get("acoustic_encoder", "xxx")
        llama_model_name = cfg.get("llama_model", "xxx")
        multi_fusion_type = cfg.get("multi_fusion_type", "attention")
        video_fusion_type = cfg.get("video_fusion_type", "qformer")
        audio_fusion_type = cfg.get("audio_fusion_type", "qformer")
        image_fusion_type = cfg.get("image_fusion_type", "token")

        frozen_video_Qformer = cfg.get("frozen_video_Qformer", False)
        frozen_video_proj = cfg.get("frozen_video_proj", False)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", False)
        frozen_audio_proj = cfg.get("frozen_audio_proj", False)
        frozen_multi_Qformer = cfg.get("frozen_multi_Qformer", False)
        frozen_multi_llama_proj = cfg.get("frozen_multi_llama_proj", False)
        frozen_llm = cfg.get("frozen_llm", False)
        lora_r = cfg.get("lora_r", 16)

        num_audio_query_token = cfg.get("num_audio_query_token", "xxx")
        num_video_query_token = cfg.get("num_video_query_token", "xxx")
        num_multi_query_token = cfg.get("num_multi_query_token", "xxx")
        num_image_query_token = cfg.get("num_image_query_token", "xxx")

        rapo_aux_loss_weight = cfg.get("rapo_aux_loss_weight", 0.2)
        rapo_conf_loss_weight = cfg.get("rapo_conf_loss_weight", 0.1)
        rapo_vocab = cfg.get("rapo_vocab", None)
        rapo_vocab_path = cfg.get("rapo_vocab_path", "")
        rapo_alias_map = cfg.get("rapo_alias_map", None)

        model = cls(
            visual_encoder_name=visual_encoder_name,
            acoustic_encoder_name=acoustic_encoder_name,
            llama_model_name=llama_model_name,
            frozen_video_proj=frozen_video_proj,
            frozen_audio_proj=frozen_audio_proj,
            frozen_multi_llama_proj=frozen_multi_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            frozen_multi_Qformer=frozen_multi_Qformer,
            frozen_llm=frozen_llm,
            lora_r=lora_r,
            num_video_query_token=num_video_query_token,
            num_audio_query_token=num_audio_query_token,
            num_multi_query_token=num_multi_query_token,
            num_image_query_token=num_image_query_token,
            multi_fusion_type=multi_fusion_type,
            video_fusion_type=video_fusion_type,
            audio_fusion_type=audio_fusion_type,
            image_fusion_type=image_fusion_type,
            rapo_aux_loss_weight=rapo_aux_loss_weight,
            rapo_conf_loss_weight=rapo_conf_loss_weight,
            rapo_vocab=rapo_vocab,
            rapo_vocab_path=rapo_vocab_path,
            rapo_alias_map=rapo_alias_map,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"], strict=False)

        ckpt_path_2 = cfg.get("ckpt_2", "")
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"], strict=False)

        ckpt_path_3 = cfg.get("ckpt_3", "")
        if ckpt_path_3:
            print("Load third Checkpoint: {}".format(ckpt_path_3))
            ckpt = torch.load(ckpt_path_3, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"], strict=False)

        return model
