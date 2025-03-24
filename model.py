import torch
import pytorch_lightning as pl
import os
from typing import Optional, Tuple, List, Union
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image, ImageOps
from torch.nn.functional import interpolate
import torch.nn.functional as F
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler, DDPMScheduler
# from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
import torch.nn as nn

dependencies = ["torch", "numpy", "diffusers", "PIL"]

from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
from stablenormal.pipeline_stablenormal import StableNormalPipeline
from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
import warnings
import math
from itertools import chain
import logging
from src.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor


def batch_pad_to_square(images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """优化的批量填充方法"""
    B, C, H, W = images.shape
    max_size = max(H, W)
    
    # 计算对称填充量
    pad_h = (max_size - H) // 2, (max_size - H + 1) // 2
    pad_w = (max_size - W) // 2, (max_size - W + 1) // 2
    
    # 使用高效填充
    return F.pad(images, (pad_w[0], pad_w[1], pad_h[0], pad_h[1])), (H, W)

def batch_resize(images: torch.Tensor, resolution: int) -> torch.Tensor:
    """批量调整大小并保持宽高比 (B, C, H, W)"""
    B, C, H, W = images.shape
    scale = resolution / min(H, W)
    
    # 向量化计算新尺寸
    new_H = (H * scale / 64).int() * 64
    new_W = (W * scale / 64).int() * 64
    
    # 使用组合插值方法
    return F.interpolate(
        images,
        size=(new_H.item(), new_W.item()),
        mode='bicubic',
        align_corners=False,
        antialias=True
    )

def center_crop(image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int, int]]:
    """Crop the center of the tensor to make it square (C, H, W)."""
    _, h, w = image.shape
    crop_size = min(h, w)
    
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    
    cropped = image[:, top:top+crop_size, left:left+crop_size]
    return cropped, (h, w), (left, top, left+crop_size, top+crop_size)

def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def scale_image(image):
    image = image * 0.5 / 0.8
    return image

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 图像投影模型：将CLIP图像特征映射为提示词序列
# 输入形状[bs clip_embed_dim] 输出形状 [bs, num_tokens, cross_attention_dim]
class ImageProjModel(torch.nn.Module):
    def __init__(self, clip_embed_dim=1024, cross_attention_dim=1024, num_tokens=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.generator = None

        self.proj = torch.nn.Linear(clip_embed_dim, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
 
    def forward(self, x):
        # image_embeds: [B, clip_embed_dim]
        assert not torch.isnan(x).any(), "输入包含NaN"

        b, c, h, w = x.shape

        # 1. 展平空间维度: [b, 1024, 2304]
        x = x.reshape(b, c, h * w)

        # 2. 交换维度和合并Batch: [b*2304, 1024]
        x = x.permute(0, 2, 1)  # [b, 2304, 1024]
        x = x.reshape(-1, c)    # [b*2304, 1024]

        # 3. 线性映射: [b*2304, cross_attention_dim]
        x = self.proj(x)
        x = x.reshape(b, h * w, -1)  # [b, 2304, cross_attention_dim]

        # 5. 归一化
        x = self.norm(x)
        return x
    
class AdapterModule(torch.nn.Module):
    def __init__(self, unet, adapter_dim=64):
        super().__init__()
        # 为UNet的每个交叉注意力层添加适配层
        self.adapters = torch.nn.ModuleList()
        
        for block in unet.down_blocks + unet.up_blocks:
            if hasattr(block, "attentions"):
                for attn in block.attentions:
                    for transformer_block in attn.transformer_blocks:

                        # 注入适配层到交叉注意力模块
                        adapter = torch.nn.Sequential(
                            torch.nn.LayerNorm(adapter_dim),
                            torch.nn.Linear(transformer_block.attn2.to_k.in_features, adapter_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(adapter_dim, transformer_block.attn2.to_k.in_features)
                        )
                        self.adapters.append(adapter)
                        # 冻结原始权重，只训练适配器
                        for param in transformer_block.attn2.parameters():
                            param.requires_grad = False
                        

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    # def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
    #     ip_tokens = self.image_proj_model(image_embeds)
    #     encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
    #     # Predict the noise residual
    #     noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
    #     return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

# 创建 LightningModule 作为封装器
class WrapperLightningModule(pl.LightningModule):
    def __init__(
        self, 
        local_cache_dir: Optional[str] = None, 
        ckpt_path:Optional[str] = None,
        device="cuda:0", 
        yoso_version='yoso-normal-v0-3', 
        diffusion_version='stable-normal-v0-1',
        drop_cond_prob = 0.05,
        prompt="The normal map",

        
        # ip_adpter_weight: Optional[str] = "ip-adapter_sd15_fixed.safetensors",
    ):
        super().__init__()

        logging.getLogger("diffusers.models.attention_processor").setLevel(logging.ERROR)  

        # 初始化配置参数
        self.local_cache_dir = local_cache_dir
        self.yoso_version = yoso_version
        self.diffusion_version = diffusion_version
        # self.ip_adpter_weight = ip_adpter_weight
        self.num_timesteps = 500
        self.drop_cond_prob = drop_cond_prob
        self.automatic_optimization = True
        self.prompt = prompt
        self.prompt_embeds = None
        torch.autograd.set_detect_anomaly(True)
        # 添加内存优化配置
        self.batch_format_ready = False
        self.best_loss = float('inf')
        
        # 缓存预处理结果
        self.preprocessed_cache = {}
        
        # 自动混合精度配置
        self.autocast_enabled = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
    

        # 初始化模型组件
        self.x_start_pipeline = YOSONormalsPipeline.from_pretrained(
            os.path.join(local_cache_dir, yoso_version),
            trust_remote_code=True,
            safety_checker=None,
            variant="fp16",
            torch_dtype=torch.float16
        ).to(device)
        
        self.pipe = StableNormalPipeline.from_pretrained(
            os.path.join(local_cache_dir, diffusion_version),
            trust_remote_code=True,
            safety_checker=None,
            variant="fp16",
            torch_dtype=torch.float16,
            scheduler=HEURI_DDIMScheduler(
                prediction_type='sample',
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear"
            ),
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )
        original_conv_in = self.pipe.unet.conv_in
        self.conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1, dtype=torch.float16)

        # 创建新的 conv_in 层（输入通道改为 8，其他参数不变）
        new_conv_in = nn.Conv2d(
            in_channels=8,  # 修改输入通道为 8
            out_channels=original_conv_in.out_channels,
            kernel_size=original_conv_in.kernel_size,
            stride=original_conv_in.stride,
            padding=original_conv_in.padding,
            dilation=original_conv_in.dilation,
            groups=original_conv_in.groups,
            bias=original_conv_in.bias is not None,
            padding_mode=original_conv_in.padding_mode
        )

        # 权重复制逻辑（核心修改部分）
        with torch.no_grad():
            # 获取原始权重 [out_channels=320, in_channels=4, kH, kW]
            original_weight = original_conv_in.weight
            
            # 沿输入通道维度复制原始权重（扩展为8通道）
            # 这里将前4通道复制到后4通道
            duplicated_weight = torch.cat([
                original_weight, 
                original_weight   # 复制并减弱新通道权重
            ], dim=1)  # 沿输入通道维度拼接
            
            # 调整权重形状以匹配新卷积层 [320, 8, 3, 3]
            new_conv_in.weight[:] = duplicated_weight
            
            # 复制偏置（若有）
            if new_conv_in.bias is not None:
                new_conv_in.bias.data = original_conv_in.bias.data.clone()


        # 替换 UNet 的 conv_in 层
        self.pipe.unet.conv_in = new_conv_in

        print(self.pipe.unet.conv_in.weight.shape)  # 应输出 torch.Size([320, 8, 3, 3])
        print(self.pipe.unet.conv_in.weight.dtype)  # 应输出 torch.float16

        train_sched = DDPMScheduler.from_config(self.pipe.scheduler.config)
        # if isinstance(self.pipe.unet, UNet2DConditionModel):
        #     self.pipe.unet = RefOnlyNoisedUNet(self.pipe.unet, train_sched, self.pipe.scheduler)

        self.train_scheduler = train_sched
        # self.unet = self.pipe.unet

        #初始化pipeline
        self.x_start_pipeline.to(device)
        self.pipe.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.pipe.scheduler.alphas_cumprod).to(device='cuda:0')
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.pipe.scheduler.alphas_cumprod).to(device='cuda:0')

        #冻结不更新参数的组件
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.sqrt_alphas_cumprod.requires_grad_(False)
        self.sqrt_one_minus_alphas_cumprod.requires_grad_(False)

        # # 在类初始化或代码开始处添加
        # warnings.filterwarnings("ignore", 
        #     message=".*cross_attention_kwargs.*are not expected by.*and will be ignored.*")

        # 加载IP适配器
        # self.pipe.load_ip_adapter(
        #     "/data/shared/TextureGeneration/IP-Adapter/IP-Adapter",
        #     subfolder="models",
        #     weight_name=ip_adpter_weight,
        #     local_files_only=True,
        #     low_cpu_mem_usage=False
        # )

        # image_proj_model = ImageProjModel(
        #     cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
        #     clip_embed_dim=1024,
        #     num_tokens=4,
        # ).float().cuda().to(device)
        image_proj_model = ImageProjModel().cuda()
        
        # init adapter modules
        attn_procs = {}
        unet_sd = self.pipe.unet.state_dict()
        for name in self.pipe.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.pipe.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipe.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                # layer_name = name.split(".processor")[0]
                # weights = {
                #     "to_k_ip.0.weight": unet_sd.get(layer_name + ".to_k.0.weight", unet_sd[layer_name + ".to_k.weight"]),
                #     "to_v_ip.0.weight": unet_sd.get(layer_name + ".to_v.0.weight", unet_sd[layer_name + ".to_v.weight"]),
                # }
                # attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                # ip_processor = IPAttnProcessor(
                #         hidden_size=hidden_size,
                #         cross_attention_dim=cross_attention_dim,
                #         num_tokens=[4],  # 兼容 num_tokens=4
                #         scale=[1.0]      # 兼容 scale=1.0
                #     )
                # # 加载权重
                # ip_processor.to_k_ip[0].weight = torch.nn.Parameter(weights["to_k_ip.0.weight"])
                # ip_processor.to_v_ip[0].weight = torch.nn.Parameter(weights["to_v_ip.0.weight"])
                # attn_procs[name] = ip_processor
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        
        self.pipe.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(self.pipe.unet.attn_processors.values()).to(device)

        # attn_processors = self.pipe.unet.attn_processors.values()
        # wrapped_processors = [AttnProcessorWrapper(proc) if not isinstance(proc, torch.nn.Module) else proc for proc in attn_processors]
        # adapter_modules = torch.nn.ModuleList(wrapped_processors).to(device)
        
        ip_adapter = IPAdapter(self.pipe.unet, image_proj_model, adapter_modules, None)

        # 步骤2: 创建IPAdapter实例
        self.image_proj_model=ip_adapter.image_proj_model
        self.adapter_modules=ip_adapter.adapter_modules

        self.image_proj_model.requires_grad_(True)
        self.adapter_modules.requires_grad_(True)
        self.conv_in.requires_grad_(True)
            
        # 日志目录
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        #生成文本嵌入
        num_images_per_prompt = 1
        with torch.no_grad():
            if self.prompt_embeds is None:
                prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                    self.prompt,
                    device,
                    num_images_per_prompt,
                    False,
                    prompt_embeds=self.prompt_embeds,
                    negative_prompt_embeds=None,
                    lora_scale=None,
                    clip_skip=None,
                )
                self.prompt_embeds = prompt_embeds
        

        # 验证可训练参数数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")  # 应大于0

        print("Modified structure:", self.pipe.unet.encoder_hid_proj)


    
    def _preprocess_image(self, size: int, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """批量预处理管道（兼容PIL和Tensor输入）"""
        # 输入格式验证
        if isinstance(images, list):
            # 情况1：输入为PIL图像列表
            tensor_batch = torch.stack([
                transforms.functional.to_tensor(img.convert('RGB')) 
                for img in images
            ]).to(self.device, non_blocking=True)
        elif isinstance(images, torch.Tensor):
            # 情况2：输入已经是预处理过的Tensor
            if images.dim() == 3:
                tensor_batch = images.unsqueeze(0).to(self.device)
            elif images.dim() == 4:
                tensor_batch = images.to(self.device)
            else:
                raise ValueError(f"Invalid tensor dimensions: {images.shape}")
        else:
            raise TypeError(f"Unsupported input type: {type(images)}")

        # 自动混合精度优化
        with torch.cuda.amp.autocast(enabled=self.autocast_enabled):
            # 批量填充 (B, C, H, W)
            padded, _ = batch_pad_to_square(tensor_batch)
            
            # 带反锯齿的调整大小
            resized = F.interpolate(
                padded,
                size=(size, size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            )
            
        # 归一化到[-1, 1]范围
        return resized.mul(2).sub(1)
    

    def _optimize_batch_format(self):
        """优化批量处理配置"""
        torch.backends.cudnn.benchmark = True
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        self.batch_format_ready = True
    

    def prepare_batch_data(self, batch):
        """
        输入数据格式:
        batch = {
            'input_images': PIL图像列表,
            'ref_normal': 提示图像列表,
            'render_gt':目标图像列表
        }
        """
        lrm_generator_input = {}
        render_gt = {}   # for supervision
        processed = {}
        

        images = self._preprocess_image(
            size=768, 
            images=batch['input_images']  # 直接传入整个图像列表
        ).to(self.device)
        
        size_crop = 512
        # 参考法线图
        if 'ref_normals' in batch and batch['ref_normals'] is not None:
            ref_normal = self._preprocess_image(
                size=size_crop,
                images=batch['ref_normals']  # 直接传递整个batch
            ).to(self.device)
        else:
            ref_normal = None


        # 处理 render_gt 字段
        if 'render_gt' in batch and batch['render_gt'] is not None:
            _render_gt = batch['render_gt']  # 直接使用原始值
            render_gt = _render_gt.to(self.device)
        else:
            render_gt = None


        lrm_generator_input['images'] = images.to(self.device)
        lrm_generator_input['ref_normals'] = ref_normal.to(self.device)

        image_resolution = 768
        init_latents = torch.zeros([1, 4, image_resolution // 8, image_resolution // 8], 
                                    device="cuda", dtype=torch.float16)
        lrm_generator_input['init_latents'] = init_latents.to(self.device)


        return lrm_generator_input, render_gt
    

    
    def training_step(self, batch, batch_idx):

        with torch.cuda.amp.autocast(enabled=self.autocast_enabled):
            model_inputs, render_gt = self.prepare_batch_data(batch)

            image = model_inputs['images']
            ref_normal = model_inputs['ref_normals']

            image = image.squeeze(1)  # 1 3 768 768
            ref_normal = ref_normal.squeeze(1)
            render_gt = render_gt.squeeze(1)

            # sample random timestep
            B = image.shape[0]
            t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

            '''
            encode_condition_image使用采样，生成带随机性latents，用于训练，增强鲁棒性
            prepare_latents使用均值，适合用于推理阶段，能生成确定性的latents
            '''

            latents, generator = None, None

            # classifier-free guidance
            if np.random.rand() < self.drop_cond_prob:
                cond_latents = self.encode_condition_image(torch.zeros_like(image))# [1, 4, 96, 96]
            else:
                cond_latents = self.encode_condition_image(image)


            latents_gt = self.encode_target_images(render_gt)
            noise = torch.randn_like(latents_gt)
            latents_noisy = self.train_scheduler.add_noise(latents_gt, noise, t)
            
            #对ref_normal使用dino encoder 生成features
            ref_normal = ref_normal.to(dtype=next(self.pipe.prior.parameters()).dtype) # bs 3 512 512
            # print(f"ref_normal:{ref_normal.shape}")
            dino_features_ref = self.pipe.prior(ref_normal)

            dino_features_ref = dino_features_ref.type(torch.float16) # bs 1024 48 48 
            # print(f"dino_features_ref:{dino_features_ref.shape}")

            #使用dino encoder的特征生成tokens
            ip_tokens = self.image_proj_model(dino_features_ref) # bs 48*48 1024
            # print(f"ip_tokens:{ip_tokens.shape}")
            ip_tokens = ip_tokens.half()
            batch_size = ip_tokens.size(0)
            # 根据bs扩展文本嵌入的维度
            self.prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1) # bs 77 1024
            encoder_hidden_states = torch.cat([self.prompt_embeds, ip_tokens], dim=1) # bs 77 1024 + bs 2304 1024

            # 内存优化：释放不需要的缓存
            torch.cuda.empty_cache()

            v_target = self.get_v(latents_gt, noise, t).half()
            
            # 使用梯度累积 
            prev_noise = self.forward(latents_noisy, cond_latents, t, encoder_hidden_states)
            loss, loss_dict = self.compute_loss(prev_noise, v_target)

            # 新增代码：记录最小loss值
            current_loss = loss.item()  # 将loss转为Python数值
            if current_loss < self.best_loss:
                old_best_loss = self.best_loss  
                self.best_loss = current_loss
                print(f"🙏🙏🙏 oh!!! New minimum loss achieved: {old_best_loss:.4f} --> {self.best_loss:.4f}")
                print(f"(↓{old_best_loss - self.best_loss:.4f})")

            # 日志记录
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        #可视化保存（每100 * accumulate_grad_batches步）
        if self.global_step % 25 == 0 and self.global_rank == 0:
            self._save_training_samples(latents_noisy, t, prev_noise, render_gt, input_image = image, ref_image = ref_normal)
            
        return loss

    
    def forward(self, latents_noisy, cond_latents, t, encoder_hidden_states):
        """优化的前向传播"""

        latent = torch.cat([latents_noisy, cond_latents], dim=1)
        # print(f"latent:{latent.shape}")

        prev_noise = self.pipe.dino_unet_forward(
            self.pipe.unet,
            latent,
            t,
            encoder_hidden_states = encoder_hidden_states,
            return_dict = False,
        )[0]

        return prev_noise


    def compute_loss(self, noise_pred, noise_gt):

        # 1. 计算MSE损失（模型预测的噪声 vs 真实噪声）
        loss = F.mse_loss(noise_pred, noise_gt)
        
        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss':loss})
        
        return loss, loss_dict


    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        # 准备验证数据
        lrm_generator_input, render_gt = self.prepare_batch_data(batch)
        
        # 生成验证结果
        # with torch.no_grad():
        #     render_out = self.forward(lrm_generator)

        render_images = render_gt['render_gt']
        self.validation_step_outputs.append(render_images)
        # 保存结果
        self._save_validation_results(render_images, batch)


    @torch.no_grad
    def _save_training_samples(self, latents_noisy, t, prev_noise, render_gt, input_image, ref_image):
        """保存训练过程样本"""
        #预测起始latents、时间步、预测噪声
        latents_pred = self.predict_start_from_z_and_v(latents_noisy, t, prev_noise)

        # 使用训练网络预测的噪声给已经加噪的latents降噪
        latents = unscale_latents(latents_pred)
        # 将 scaling_factor 转换为张量，并确保类型和设备匹配
        self.pipe.vae.config.scaling_factor = torch.tensor(
            self.pipe.vae.config.scaling_factor,
            dtype=torch.float16,  # 强制使用 float16
            device=latents.device
        )

        # 确保 latents 是 float16
        latents = latents.to(torch.float16)

        prediction = unscale_image(self.pipe.decode_prediction(latents))
        images = (prediction.clip(-1, 1) + 1) / 2

        # images = unscale_image(self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
        # images = (images * 0.5 + 0.5).clamp(0, 1)
        ref_image = F.interpolate(ref_image, size=(768, 768), mode='bilinear')
        group1 = torch.cat([ref_image, input_image], dim=-1) 
        group2 = torch.cat([render_gt, images], dim=-1)
        images = torch.cat([group1, group2], dim=-2)

        grid = make_grid(images, nrow=images.shape[0], normalize=True, value_range=(0, 1))

        save_path = os.path.join(self.logdir, 'images', f"*train_step_{self.global_step:07d}.png")
        save_dir = os.path.join(self.logdir, 'images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_image(grid, save_path, normalize=True)
        


    def _save_validation_results(self, output, batch):
        """保存验证结果"""
        val_dir = os.path.join(self.log_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        
        for i, (img, normal) in enumerate(zip(batch['input_images'], output['generated_normals'])):
            save_image(normal, os.path.join(val_dir, f"val_{self.current_epoch}_{i}.png"))


    def encode_condition_image(self, images):
        dtype = next(self.pipe.vae.parameters()).dtype
        
        # 硬编码CLIP标准归一化参数
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                            device=self.device).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=self.device).view(1, 3, 1, 1)
        
        # 直接处理已经预处理好的张量
        # 假设输入images满足：
        # 1. 已经是RGB三通道 (无alpha通道)
        # 2. 尺寸已调整为224x224
        # 3. 数值范围在[0,1]之间
        
        # 执行CLIP特有的归一化
        normalized = (images.to(self.device) - clip_mean) / clip_std
        
        # 编码为潜在空间
        latents = self.pipe.vae.encode(normalized.to(dtype)).latent_dist.sample()
        return latents

    def _get_resample_mode(self, resample_id: int) -> str:
        """将resample参数转换为PyTorch插值模式"""
        resample_map = {
            0: "nearest",
            2: "bilinear",
            3: "bicubic"
        }
        return resample_map.get(resample_id, "bicubic")

    def configure_optimizers(self):
        # 添加自动学习率缩放
        base_lr = self.learning_rate
        optimizer = torch.optim.AdamW(
            params=chain(
                self.image_proj_model.parameters(),
                self.adapter_modules.parameters(),
                ),
            lr=base_lr,
            weight_decay=1e-2,
            fused=True  # 启用融合优化器
        )
        
        # 动态调整学习率
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0 = 500,
                T_mult = 2,
                eta_min = base_lr / 4,
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
    



    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipe.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8   # [-0.625, 0.625]
        posterior = self.pipe.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents
    

    def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))


    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )


    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    
