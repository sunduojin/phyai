#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    # 这两个字段决定模型主体用哪一档 Gemma 配置。
    # pi0 由两套权重协同工作：
    # 1. paligemma_variant: 视觉-语言主干，处理图像 token 和语言 token。
    # 2. action_expert_variant: 动作专家，处理机器人 state 和 noisy action chunk。
    # 默认配置对应常见的 pi0 结构：Gemma 2B 级别的 VLM + Gemma 300M 级别的 action expert。
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    # 模型参数/激活的主要精度选项。
    # float32 更稳、更容易 debug；bfloat16 更省显存，通常用于大模型推理或训练。
    # 注意：这里不是 PyTorch dtype 对象，而是字符串，后面会在 modeling_pi0.py 里解析。
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    # pi0 当前只使用当前时刻的一帧观测，所以 n_obs_steps=1。
    # 有些策略会看历史多帧，例如 diffusion policy 可能用过去几帧做条件；
    # pi0 这里的输入是当前图像、当前语言指令、当前机器人 state。
    n_obs_steps: int = 1

    # chunk_size 是模型一次预测的动作序列长度，也就是 action horizon。
    # 在 modeling_pi0.py 里，sample_actions() 会生成形状类似 (B, chunk_size, action_dim) 的动作块。
    # pi0 默认一次预测未来 50 步动作，而不是只预测下一步动作。
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"

    # n_action_steps 是真正拿去执行/入队的动作步数。
    # 如果 n_action_steps < chunk_size，模型仍然预测完整 chunk，但 policy 只执行前面一部分。
    # select_action() 里有 action queue：队列空时预测一个 chunk，然后逐步弹出动作。
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    # 不同机器人有不同的 state/action 维度，例如单臂、双臂、夹爪数量都可能不同。
    # pi0 用统一的最大维度承载它们：真实维度不足时 pad 到 max_state_dim / max_action_dim。
    # 这样同一个模型结构可以兼容多个机器人形态。
    # 在 PI0Pytorch 中：
    # - state 会通过 state_proj: Linear(max_state_dim -> expert_width)
    # - action 会通过 action_in_proj/action_out_proj 在 max_action_dim 和 expert_width 间转换
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    # flow matching 是 pi0 生成连续动作的核心。
    # 训练时：把真实动作 A 和高斯噪声 eps 按时间 t 插值，得到 noisy action；
    #        模型学习从 noisy action 指向真实动作的速度场。
    # 推理时：从纯噪声开始，做 num_inference_steps 次 Euler 更新，逐步得到动作 chunk。
    num_inference_steps: int = 10  # Number of denoising steps during inference

    # 训练时采样 flow matching 时间 t 的 Beta 分布参数。
    # modeling_pi0.py 中 sample_time() 会先从 Beta(alpha, beta) 采样，
    # 再用 scale/offset 把 t 限制在 (0, 1) 附近的稳定范围内。
    # 直觉上：不同的 t 表示 noisy action 处在“更像噪声”还是“更像真实动作”的位置。
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0

    # 避免 t 精确落到 0 或 1。
    # t=0/1 的极端点有时会带来数值或训练稳定性问题，所以 OpenPI/LeRobot 使用
    # time = beta_sample * scale + offset，让 t 大致落在 [0.001, 1.0]。
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    # 时间 t 会先被编码成正弦位置嵌入，再与 action embedding 融合。
    # min_period/max_period 控制这组 sin/cos 特征覆盖的频率范围。
    # 可以把它理解成“让网络用多种频率感知当前 denoise 进度”。
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    # 是否把数据集里的 absolute action 转成 relative action。
    # 对机器人来说，relative action 常表示“相对当前关节/末端位姿的增量”，
    # 有时比直接预测绝对位置更容易泛化。
    # processor_pi0.py 里会先做 relative conversion，再做 normalization。
    use_relative_actions: bool = False

    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    # 有些关节不适合转成相对量，典型例子是 gripper。
    # gripper 往往更像开/合或目标宽度，而不是连续位姿增量，所以默认排除。
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])

    # Populated at runtime from dataset metadata by make_policy.
    # 动作维度对应的名字，例如每个 joint/gripper 的名称。
    # 这个字段通常不是手写配置，而是在创建 policy 时根据数据集 metadata 填充，
    # RelativeActionsProcessorStep 会用它判断哪些维度需要排除。
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    # RTC 是推理阶段的平滑/实时执行增强。
    # pi0 一次输出一个长 action chunk，但机器人控制是逐步执行的；
    # RTC 可以利用上一个 chunk 和当前 chunk 的关系，让慢模型的动作执行更平滑。
    rtc_config: RTCConfig | None = None

    # PaliGemma/SigLIP 默认使用 224x224 图像。
    # 这里必须是正方形，因为 modeling_pi0.py 里会检查高宽相等；
    # 图像预处理会把相机图像 resize/crop 到这个分辨率后送入 vision tower。
    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    # pi0/PaliGemma 通常假设有固定数量的图像输入。
    # 如果某个数据集相机数少于模型期望，可以补空相机占位，保持输入 key/shape 对齐。
    # validate_features() 里会根据 empty_cameras 动态加入 observation.images.empty_camera_i。
    empty_cameras: int = 0

    # Normalization
    # 归一化策略告诉 processor 如何处理不同类型的输入/输出。
    # VISUAL 使用 IDENTITY：图像通常在图像 processor/模型内部处理，不走数据集统计归一化。
    # STATE/ACTION 使用 MEAN_STD：机器人状态和动作按数据集均值方差标准化。
    # 注意：pi0 和 pi05 不完全一样，pi05 默认用 QUANTILES；pi0 这里是 MEAN_STD。
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    # gradient_checkpointing 用计算换显存：
    # 前向时不保存部分中间激活，反向时再重算，适合显存紧张的大模型训练。
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization

    # 是否用 torch.compile 编译模型。
    # 对大模型推理/训练可能有加速，但首次编译慢，且 debug 难度会增加。
    compile_model: bool = False  # Whether to use torch.compile for model optimization

    # torch.compile 的模式。"max-autotune" 往往更激进，可能带来更好性能，
    # 但编译开销也更高。调试阶段通常先保持 compile_model=False。
    compile_mode: str = "max-autotune"  # Torch compile mode

    # 指定运行设备，例如 "cuda"、"cpu"、"mps"。
    # None 表示由上层逻辑自动选择；processor_pi0.py 也会用这个字段把 batch 移到对应设备。
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    # 只冻结视觉编码器。适合数据少、只想让语言/action 侧适配新任务的场景。
    # 视觉塔通常已经在大规模图文数据上预训练好，直接全量微调容易显存高且过拟合。
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder

    # 冻结整个 VLM，只训练 action expert 和动作相关投影层。
    # 对机器人小数据集微调很常见：保留视觉语言先验，只让机器人动作头学新 embodiment。
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW``
    # AdamW 是训练/微调 pi0 的默认优化器配置。
    # lr/betas/eps/weight_decay 都会在 get_optimizer_preset() 里打包成 AdamWConfig。
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01

    # 梯度裁剪阈值，防止大模型训练中偶发梯度爆炸。
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    # 学习率调度：先 warmup 到 optimizer_lr，再用 cosine decay 降到 scheduler_decay_lr。
    # 如果总训练步数比 scheduler_decay_steps 小，上层训练逻辑会按实际步数缩放。
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # 语言 tokenizer 的最大长度。
    # pi0 的语言输入主要是 task prompt，不像 pi05 那样把 state 离散化拼进文本，
    # 所以默认 48 比 pi05 的 200 短很多。
    # processor_pi0.py 的 TokenizerProcessorStep 会用这个值 padding/truncation。
    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    def __post_init__(self):
        # 先让 PreTrainedConfig 做通用初始化/校验，例如处理 input_features/output_features。
        super().__post_init__()

        # Validate configuration
        # 模型一次最多只预测 chunk_size 步，所以真正执行的步数不能超过预测长度。
        # 否则 action queue 会尝试取出模型没有预测的动作。
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        # 当前实现只支持这两档 Gemma 配置。
        # 如果要接别的尺寸，除了这里放开，还需要确认 modeling_pi0.py 中 get_gemma_config()
        # 和权重加载逻辑都有对应实现。
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        # action expert 也只支持这两档 Gemma 配置。
        # pi0 默认用较小的 gemma_300m，因为动作 token 序列比语言模型任务更专门，
        # 同时这样能显著降低计算和显存压力。
        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        # dtype 字段后面会按字符串分支处理，所以这里尽早发现拼写错误。
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features.

        这个函数把“模型默认需要哪些输入/输出”补进 config。
        LeRobot 的数据集和 processor 都依赖 input_features/output_features 知道：
        - 哪些 key 是图像、state、action；
        - 每个 key 的 shape 是多少；
        - 哪些字段需要归一化或反归一化。
        """
        for i in range(self.empty_cameras):
            # 给缺失相机补占位图像。这样下游模型仍然能按固定相机列表取图像。
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            # 如果用户/数据集没有显式声明 observation.state，就补一个默认 state 输入。
            # 真实 state 维度较短时，会在模型侧 pad 到 max_state_dim。
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            # 如果没有显式声明 action 输出，就补一个默认 action feature。
            # 模型输出宽度是 max_action_dim；实际机器人只使用对应的真实动作维度。
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        # 训练脚本会调用这个方法拿到默认 AdamW 配置。
        # 这样命令行只指定 policy.type=pi0 时，也能获得一套和 OpenPI 接近的优化器参数。
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        # 返回默认学习率调度器：warmup + cosine decay。
        # peak_lr 对应 optimizer_lr，最终衰减到 scheduler_decay_lr。
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        # 告诉数据集采样器：pi0 不需要历史观测帧，只取当前时刻观测。
        # 返回 None 表示不额外构造 observation 的时间偏移序列。
        return None

    @property
    def action_delta_indices(self) -> list:
        # 告诉数据集采样器：训练时需要当前时刻开始的 chunk_size 个未来动作。
        # 这些未来动作会组成监督信号 action chunk，形状通常是 (chunk_size, action_dim)。
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        # pi0 是行为克隆/flow matching 策略，不直接使用 reward 序列。
        return None
