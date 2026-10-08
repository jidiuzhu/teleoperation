# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
# Heavy inspiration taken from
# * DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# * DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT
# * DiT Policy by Dasari et. al. : https://github.com/sudeepdasari/dit-policy

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import warnings
from collections import deque
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
    ACTION,
    OBS_IMAGES,
)
from lerobot.configs.types import FeatureType
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from .configuration_ditflow import BiFlowVLAConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)


def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class _TimeNetwork(nn.Module):
    def __init__(
        self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000
    ):
        assert frequency_embedding_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(frequency_embedding_dim // 2)
        super().__init__()

        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        assert len(t.shape) == 1, "assumes 1d input timestep array"
        t = t[:, None] * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)
        return self.out_net(t)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create mlp
        self.mlp = nn.Sequential(
            self.linear1,
            self.activation,
            self.dropout2,
            self.linear2,
            self.dropout3,
        )

        # create modulation layers
        self.attn_modulate = _ShiftScaleMod(d_model)
        self.attn_gate = _ZeroScaleMod(d_model)
        self.mlp_modulate = _ShiftScaleMod(d_model)
        self.mlp_gate = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond, need_weights=False):
        # process the conditioning vector first
        cond = cond + t

        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=need_weights)
        x = x + self.attn_gate(self.dropout1(x2), cond)

        x3 = self.mlp_modulate(self.norm2(x), cond)
        x3 = self.mlp(x3)
        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )
        self.reset_parameters()

    def forward(self, src, t, cond, **kwargs):
        x = src
        for layer in self.layers:
            x = layer(x, t, cond, **kwargs)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class _YDiTNoiseNet(nn.Module):
    def __init__(
        self,
        wrist_dim,
        num_tokens,
        vocab_size,
        token_emb_dim,
        cond_dim,
        horizon=64,
        time_dim=256,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
        clip_sample=False,
        clip_sample_range=1.0,
    ):
        super().__init__()
        self.wrist_dim = wrist_dim
        self.num_tokens = num_tokens
        self.vocab_size = vocab_size
        
        # 🌟 真实动作输入维度：XYZ+6D(9) + 8*64(512) = 521
        self.ac_dim = wrist_dim + (num_tokens * token_emb_dim)
        self.ac_chunk = horizon

        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(self.ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.ac_proj = nn.Sequential(
            nn.Linear(self.ac_dim, self.ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ac_dim, hidden_dim),
        )

        decoder_module = _DiTDecoder(
            hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # ==========================================
        # 🌟 核心创新：Y-Shape 解耦输出头
        # ==========================================
        # 分支 A: 预测连续手腕流 (Flow Matching) -> 输出 [B, T, 9]
        self.arm_head = _FinalLayer(hidden_dim, wrist_dim)
        # 分支 B: 预测离散手指意图的 Logits -> 输出 [B, T, 8 * 512]
        self.hand_head = _FinalLayer(hidden_dim, num_tokens * vocab_size)

        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

    def forward(self, noisy_actions, time, global_cond, need_weights=False):
        c = self.cond_proj(global_cond)
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noisy_actions).transpose(0, 1)  # [T, B, H]
        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)]

        # DiT 主干传播
        dec_out = self.decoder(dec_in, time_enc, c, need_weights=need_weights)

        # 🌟 Y-Shape 双路输出
        eps_wrist = self.arm_head(dec_out, time_enc, c).transpose(0, 1)  # [B, T, wrist_dim]
        hand_logits = self.hand_head(dec_out, time_enc, c).transpose(0, 1)  # [B, T, num_tokens * vocab_size]
        
        # Reshape logits 便于后续算交叉熵
        B, T = hand_logits.shape[:2]
        hand_logits = hand_logits.view(B, T, self.num_tokens, self.vocab_size)  # [B, T, 8, 512]
        
        return eps_wrist, hand_logits

    def sample_noise(self, batch_size: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(
            batch_size, self.ac_chunk, self.ac_dim, device=device, generator=generator
        )

class BiFlowVLAPolicy(PreTrainedPolicy):
    """
    BiFlow VLA Policy: Y-Shape dual-branch flow matching VLA policy
    combining a frozen VLM (Qwen3-VL) with a DiT-based action expert
    for simultaneous continuous wrist trajectory and discrete finger token prediction.
    """

    config_class = BiFlowVLAConfig
    name = "biflow_vla"

    def __init__(
        self,
        config: BiFlowVLAConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.dit_flow = BiFlowVLAModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.dit_flow.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            # 🌟 把原本单一的 action 队列，拆成手腕和手指两个队列
            "action.wrist_traj": deque(maxlen=self.config.n_action_steps),
            "action.hand_tokens": deque(maxlen=self.config.n_action_steps),
        }

        # Add queues for all other STATE features (e.g., observation.velocity)
        for key, feature in self.config.input_features.items():
            if feature.type is FeatureType.STATE and key != "observation.state":
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)
        
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps
            )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> dict:
        batch = {
            k: torch.stack(list(self._queues[k]), dim=1)
            for k in batch
            if k in self._queues
        }
        
        # 获取条件和采样
        global_cond = self.dit_flow._prepare_global_conditioning(batch)
        actions_dict = self.dit_flow.conditional_sample(
            batch_size=global_cond.shape[0], global_cond=global_cond
        )

        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        
        # 截取有效长度
        return {
            "wrist_traj": actions_dict["wrist_traj"][:, start:end],
            "hand_tokens": actions_dict["hand_tokens"][:, start:end]
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> dict:
        # 清理无用 action
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
            
        self._queues = populate_queues(self._queues, batch)

        # 🌟 如果队列空了，就调用网络推理一次 Action Chunk
        if len(self._queues["action.wrist_traj"]) == 0:
            actions_dict = self.predict_action_chunk(batch)
            
            # 把预测出的 chunk 放进队列
            wrist_chunk = actions_dict["wrist_traj"][0]   # [T, 9]
            token_chunk = actions_dict["hand_tokens"][0]  # [T, 8]
            for i in range(wrist_chunk.shape[0]):
                self._queues["action.wrist_traj"].append(wrist_chunk[i])
                self._queues["action.hand_tokens"].append(token_chunk[i])

        # 🌟 每次从队列中弹出一帧动作执行
        return {
            "action.wrist_traj": self._queues["action.wrist_traj"].popleft(),
            "action.hand_tokens": self._queues["action.hand_tokens"].popleft()
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        loss = self.dit_flow.compute_loss(batch)
        return loss, None

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class BiFlowVLAModel(nn.Module):
    def __init__(self, config: BiFlowVLAConfig):
        super().__init__()
        self.config = config

        # ==========================================
        # 🌟 1. 引入并冻结 Qwen-VL (The Eyes)
        # ==========================================
        # 初始化 Qwen-VL (训练时如果显存不够，可配合 PEFT/LoRA)
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_model_id, torch_dtype=torch.bfloat16
        )
        self.vlm_processor = AutoProcessor.from_pretrained(config.vlm_model_id)

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen3-VL-2B-Instruct",
        #     dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )
        # 彻底冻结视觉语言模型，防止 Few-Shot 时灾难性遗忘
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.vlm.eval()

        # 🌟 2. 初始化 Token Embedder
        # 我们之前把 token_emb_dim 设为 64，就是为了这里！
        # 之后你可以用预训练 VQ-VAE 的 codebook 权重覆盖它：self.token_embedder.weight.data.copy_(...)
        self.token_embedder = nn.Embedding(config.vocab_size, config.token_emb_dim)
        self._init_token_embedder_from_vqvae()
        if config.freeze_token_embedder:
            self.token_embedder.weight.requires_grad = False

        self.vlm_feature_projector = None

        # 🌟 3. 计算全局条件维度
        # VLM特征(1536) + 腕部状态(9) + 物理本体向量(26)
        self.global_cond_dim = config.vlm_feature_dim + config.wrist_dim + config.physics_dim

        # 🌟 4. 初始化你的 Y-Shape DiT
        self.velocity_net = _YDiTNoiseNet(
            wrist_dim=config.wrist_dim,
            num_tokens=config.num_tokens,
            vocab_size=config.vocab_size,
            token_emb_dim=config.token_emb_dim,
            cond_dim=self.global_cond_dim * config.n_obs_steps,
            horizon=config.horizon,
            time_dim=config.frequency_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.num_heads,
            activation=config.activation,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )

        self.num_inference_steps = config.num_inference_steps or 10
        self.training_noise_sampling = config.training_noise_sampling
        
        if config.training_noise_sampling == "uniform":
            self.noise_distribution = torch.distributions.Uniform(low=0, high=1)
        elif config.training_noise_sampling == "beta":
            s = 0.999
            beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
            affine_transform = torch.distributions.transforms.AffineTransform(loc=s, scale=-s)
            self.noise_distribution = torch.distributions.TransformedDistribution(beta_dist, [affine_transform])
        else:
            raise ValueError(f"Unknown {config.training_noise_sampling=}")

    def _init_token_embedder_from_vqvae(self) -> None:
        if self.config.vqvae_codebook_path is None:
            return

        checkpoint = torch.load(self.config.vqvae_codebook_path, map_location="cpu")
        state_dict: dict[str, Any]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported VQ-VAE checkpoint format; expected a dict-like checkpoint.")

        codebook_weight = None
        for key in self.config.vqvae_codebook_key_candidates:
            if key in state_dict:
                codebook_weight = state_dict[key]
                break

        if codebook_weight is None:
            raise ValueError(
                f"Cannot find VQ-VAE codebook weight in checkpoint. Tried keys: {self.config.vqvae_codebook_key_candidates}"
            )

        if not torch.is_tensor(codebook_weight):
            codebook_weight = torch.tensor(codebook_weight)

        expected_shape = (self.config.vocab_size, self.config.token_emb_dim)
        if tuple(codebook_weight.shape) != expected_shape:
            raise ValueError(
                f"VQ-VAE codebook shape mismatch. Got {tuple(codebook_weight.shape)}, expected {expected_shape}."
            )

        with torch.no_grad():
            self.token_embedder.weight.copy_(codebook_weight)

    def _extract_vlm_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if OBS_IMAGES not in batch:
            raise ValueError(f"Expected `{OBS_IMAGES}` in batch for Qwen3-VL conditioning.")

        obs_images = batch[OBS_IMAGES]
        # [B, T, N, C, H, W] -> pick first camera: [B, T, C, H, W]
        if obs_images.dim() == 6:
            obs_images = obs_images[:, :, 0]
        if obs_images.dim() != 5:
            raise ValueError(f"Unexpected image tensor shape: {tuple(obs_images.shape)}")

        B, T = obs_images.shape[:2]
        device = obs_images.device

        # Flatten to per-frame and convert to uint8 HWC for processor
        flat = obs_images.reshape(B * T, *obs_images.shape[2:])
        flat = flat.detach().float()
        # Robust rescale for possibly normalized inputs
        flat_min = flat.amin(dim=(1, 2, 3), keepdim=True)
        flat_max = flat.amax(dim=(1, 2, 3), keepdim=True)
        flat = (flat - flat_min) / (flat_max - flat_min + 1e-6)
        flat_uint8 = (flat.clamp(0, 1) * 255.0).to(torch.uint8)
        np_images = [img.permute(1, 2, 0).cpu().numpy() for img in flat_uint8]

        # Use a fixed prompt so Qwen3-VL returns meaningful hidden states from image-conditioned tokens.
        prompts = [self.config.vlm_prompt] * (B * T)
        proc_inputs = self.vlm_processor(
            text=prompts,
            images=np_images,
            padding=True,
            return_tensors="pt",
        )

        vlm_device = next(self.vlm.parameters()).device
        vlm_dtype = next(self.vlm.parameters()).dtype
        proc_inputs = {
            k: (v.to(device=vlm_device, dtype=vlm_dtype) if torch.is_floating_point(v) else v.to(vlm_device))
            for k, v in proc_inputs.items()
        }

        with torch.no_grad():
            outputs = self.vlm(**proc_inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden.mean(dim=1)

        # Project to configured feature dim and reshape back to [B, T, D]
        if self.vlm_feature_projector is None:
            self.vlm_feature_projector = nn.Linear(
                pooled.shape[-1],
                self.config.vlm_feature_dim,
                bias=False,
            ).to(vlm_device)
            if not self.config.vlm_trainable_projector:
                for param in self.vlm_feature_projector.parameters():
                    param.requires_grad = False

        proj = self.vlm_feature_projector.to(vlm_device)(pooled)
        return proj.to(device=device, dtype=torch.float32).view(B, T, -1)

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """提取 VLM 特征，并拼接手腕状态和物理特征"""
        B, n_obs_steps = batch["observation.state"].shape[:2]
        device = batch["observation.state"].device

        try:
            vlm_context = self._extract_vlm_context(batch)
        except Exception as e:
            warnings.warn(
                f"Qwen3-VL feature extraction failed, fallback to zeros. Reason: {e}",
                RuntimeWarning,
            )
            vlm_context = torch.zeros((B, n_obs_steps, self.config.vlm_feature_dim), device=device, dtype=torch.float32)

        # 拼接: Qwen特征(1536) + 当前腕部状态(9) + 机器人物理向量(26)
        wrist_state = batch["observation.state"] # [B, n_obs_steps, 9]
        if "observation.physics_vec" in batch:
            physics_vec = batch["observation.physics_vec"]
        else:
            physics_vec = torch.zeros(
                (B, n_obs_steps, self.config.physics_dim),
                device=device,
                dtype=wrist_state.dtype,
            )
        
        global_cond = torch.cat([vlm_context, wrist_state, physics_vec], dim=-1) # [B, n_obs_steps, cond_dim]
        return global_cond.flatten(start_dim=1) # [B, cond_dim * n_obs_steps]

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """核心：计算双流 Y-Shape Loss"""
        B, T = batch["action.wrist_traj"].shape[:2]
        device = batch["action.wrist_traj"].device
        
        gt_wrist = batch["action.wrist_traj"]  # [B, T, 9]
        gt_tokens = batch["action.hand_tokens"].long() # [B, T, 8]

        global_cond = self._prepare_global_conditioning(batch)

        # 🌟 Flow Matching 时间步采样
        timesteps = self.noise_distribution.sample((B,)).to(device)
        t_expand = timesteps[:, None, None]

        # 🌟 分支A加噪: 腕部连续流
        noise_wrist = torch.randn_like(gt_wrist)
        noisy_wrist = (1 - t_expand) * noise_wrist + t_expand * gt_wrist
        target_wrist_flow = gt_wrist - noise_wrist

        # 🌟 分支B加噪: 手指离散嵌入流
        clean_hand_embeds = self.token_embedder(gt_tokens).view(B, T, -1) # [B, T, 8*64]
        noise_hand = torch.randn_like(clean_hand_embeds)
        noisy_hand_embeds = (1 - t_expand) * noise_hand + t_expand * clean_hand_embeds

        # 拼接喂给 DiT
        noisy_actions = torch.cat([noisy_wrist, noisy_hand_embeds], dim=-1)

        # 前向传播
        pred_wrist_flow, hand_logits = self.velocity_net(
            noisy_actions=noisy_actions, time=timesteps, global_cond=global_cond
        )

        # 🌟 双头 Loss
        # 1. 连续流匹配 (MSE)
        loss_arm = F.mse_loss(pred_wrist_flow, target_wrist_flow, reduction="none")
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            loss_arm = loss_arm * in_episode_bound.unsqueeze(-1)
        loss_arm = loss_arm.mean()

        # 2. 离散意图 (Cross Entropy)
        hand_logits_flat = hand_logits.view(-1, self.config.vocab_size)
        gt_tokens_flat = gt_tokens.view(-1)
        loss_hand = F.cross_entropy(hand_logits_flat, gt_tokens_flat, reduction="none")
        
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            loss_hand = loss_hand.view(B, T, self.config.num_tokens)
            loss_hand = loss_hand * in_episode_bound.unsqueeze(-1)
        loss_hand = loss_hand.mean()

        # 打印一下看看分布，方便调参 (1:1 直接相加通常是好的起点)
        total_loss = loss_arm + loss_hand
        return total_loss

    @torch.no_grad()
    def conditional_sample(self, batch_size: int, global_cond: torch.Tensor, generator=None) -> dict:
        """推理时的 ODE 求解器"""
        device = global_cond.device
        
        x_wrist = torch.randn(batch_size, self.config.horizon, self.config.wrist_dim, device=device)
        x_hand_embeds = torch.randn(batch_size, self.config.horizon, self.config.num_tokens * self.config.token_emb_dim, device=device)
        
        dt = 1.0 / self.num_inference_steps
        t_all = torch.arange(self.num_inference_steps, device=device).float().unsqueeze(0).expand(batch_size, -1) / self.num_inference_steps

        for k in range(self.num_inference_steps):
            t = t_all[:, k]
            noisy_actions = torch.cat([x_wrist, x_hand_embeds], dim=-1)
            v_wrist, _ = self.velocity_net(noisy_actions, t, global_cond)
            x_wrist = x_wrist + dt * v_wrist

        # 最后一步获取类别
        noisy_actions = torch.cat([x_wrist, x_hand_embeds], dim=-1)
        _, final_logits = self.velocity_net(noisy_actions, torch.ones_like(t_all[:, 0]), global_cond)
        pred_tokens = torch.argmax(final_logits, dim=-1) # [B, T, 8]

        return {"wrist_traj": x_wrist, "hand_tokens": pred_tokens}