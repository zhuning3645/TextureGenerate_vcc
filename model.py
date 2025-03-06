import torch
import pytorch_lightning as pl
import os
from typing import Optional, Tuple
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image, ImageOps
from torch.nn.functional import interpolate
import torch.nn.functional as F
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler, DDPMScheduler
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
dependencies = ["torch", "numpy", "diffusers", "PIL"]

from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
from stablenormal.pipeline_stablenormal import StableNormalPipeline
from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
from torchvision.transforms import v2
import warnings



def pad_to_square(image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int, int]]:
    """Pad the input tensor to make it square (C, H, W)."""
    _, h, w = image.shape
    size = max(h, w)
    
    delta_w = size - w
    delta_h = size - h
    
    # Padding format: (left, right, top, bottom)
    padding = (delta_w // 2, delta_w - (delta_w // 2),
               delta_h // 2, delta_h - (delta_h // 2))
    
    padded = F.pad(image, padding)
    return padded, (h, w), (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)

def resize_image(image: torch.Tensor, resolution: int) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[float, float]]:
    """Resize tensor while maintaining aspect ratio and pad to multiple of 64 (C, H, W)."""
    _, h, w = image.shape
    min_dim = min(h, w)
    scale = resolution / min_dim
    
    # Calculate new dimensions
    new_h = int(round(h * scale / 64)) * 64
    new_w = int(round(w * scale / 64)) * 64

    # Resize with bicubic interpolation
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)
    
    return resized, (h, w), (new_h/h, new_w/w)

def center_crop(image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int, int]]:
    """Crop the center of the tensor to make it square (C, H, W)."""
    _, h, w = image.shape
    crop_size = min(h, w)
    
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    
    cropped = image[:, top:top+crop_size, left:left+crop_size]
    return cropped, (h, w), (left, top, left+crop_size, top+crop_size)



def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

# 图像投影模型：将CLIP图像特征映射为提示词序列
class ImageProjModel(torch.nn.Module):
    def __init__(self, clip_embed_dim=1024, cross_attention_dim=1024, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        # self.proj = torch.nn.Linear(clip_embed_dim, cross_attention_dim * num_tokens)
        # self.norm = torch.nn.LayerNorm(cross_attention_dim)
        # 修改1：使用 Xavier 初始化并降低增益 (gain=0.1)
        self.proj = torch.nn.Linear(clip_embed_dim, cross_attention_dim * num_tokens).float()
        torch.nn.init.kaiming_normal_(self.proj.weight, mode='fan_out',nonlinearity='relu')  # 缩小初始权重范围
        torch.nn.init.zeros_(self.proj.bias)  # 初始化偏置为0
        
        # 修改2：在归一化前添加激活函数（如 GELU）
        self.act = torch.nn.GELU()  # 防止数值线性增长
        self.norm = torch.nn.LayerNorm(cross_attention_dim, eps = 1e-3)

        self.proj_norm = torch.nn.LayerNorm(cross_attention_dim * num_tokens)

        self.residual_norm = torch.nn.LayerNorm(cross_attention_dim)
        # 定义用于调整残差路径维度的线性变换
        self.residual_proj = nn.Linear(1024, 4096)

  

        
    def forward(self, image_embeds):
        # image_embeds: [B, clip_embed_dim]

        assert not torch.isnan(image_embeds).any(), "输入包含NaN"

        residual = self.residual_norm(image_embeds)

        print(f"image_embeds: {image_embeds.shape}")# 1 1024

        print(residual.shape) # 1 1024
        # 报错的地方，试了残差结构还是没办法解决

        x = self.proj(residual)  # [B, cross_attention_dim * num_tokens]
        x = self.proj_norm(x)
        print(x.shape) # 1 4096
        residual_transformed = self.residual_proj(residual)
        x += residual_transformed
        
        print(f"proj output x type: {x.dtype}")
        #判断NaN断言
        assert torch.isnan(x).sum() == 0, print(x)

        x = self.act(x)

        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)  # [B, num_tokens, cross_attention_dim]
        #print(f"act output max/min: {x.max()}, {x.min()}")

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
                            torch.nn.Linear(transformer_block.attn2.to_k.in_features, adapter_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(adapter_dim, transformer_block.attn2.to_k.in_features)
                        )
                        self.adapters.append(adapter)
                        # 冻结原始权重，只训练适配器
                        for param in transformer_block.attn2.parameters():
                            param.requires_grad = False
                        


class AttnProcessorWrapper(torch.nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def forward(self, *args, **kwargs):
        return self.processor(*args, **kwargs)


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred


# 创建 LightningModule 作为封装器
class WrapperLightningModule(pl.LightningModule):
    def __init__(
        self, 
        #feature_extractor_vae: CLIPImageProcessor,
        local_cache_dir: Optional[str] = None, 
        ckpt_path:Optional[str] = None,
        device="cuda:0", 
        yoso_version='yoso-normal-v0-3', 
        diffusion_version='stable-normal-v0-1',
        drop_cond_prob = 0.1,
        prompt="The normal map",

        
        # ip_adpter_weight: Optional[str] = "ip-adapter_sd15_fixed.safetensors",
    ):
        super().__init__()
            
         # 初始化配置参数
        self.local_cache_dir = local_cache_dir
        self.yoso_version = yoso_version
        self.diffusion_version = diffusion_version
        # self.ip_adpter_weight = ip_adpter_weight
        self.num_timesteps = 50
        self.drop_cond_prob = drop_cond_prob
        #self.feature_extractor_vae = feature_extractor_vae
        self.automatic_optimization = True
        self.prompt = prompt
        self.prompt_embeds = None
        torch.autograd.set_detect_anomaly(True)

        


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

        self.x_start_pipeline.to(device)
        self.pipe.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.pipe.scheduler.alphas_cumprod).to(device='cuda:0')
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.pipe.scheduler.alphas_cumprod).to(device='cuda:0')


        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.sqrt_alphas_cumprod.requires_grad_(False)
        self.sqrt_one_minus_alphas_cumprod.requires_grad_(False)

        # 在类初始化或代码开始处添加
        warnings.filterwarnings("ignore", 
            message=".*cross_attention_kwargs.*are not expected by.*and will be ignored.*")
        #self.pipe.text_encoder.requires_grad_(False)
        #self.pipe.image_encoder.requires_grad_(False)
        #self.prior = self.pipe.prior

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
        #model = ImageProjModel().float().cuda()
        

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
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.0.weight": unet_sd.get(layer_name + ".to_k.0.weight", unet_sd[layer_name + ".to_k.weight"]),
                    "to_v_ip.0.weight": unet_sd.get(layer_name + ".to_v.0.weight", unet_sd[layer_name + ".to_v.weight"]),
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                ip_processor = IPAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=[4],  # 兼容 num_tokens=4
                        scale=[1.0]      # 兼容 scale=1.0
                    )
                # 加载权重
                ip_processor.to_k_ip[0].weight = torch.nn.Parameter(weights["to_k_ip.0.weight"])
                ip_processor.to_v_ip[0].weight = torch.nn.Parameter(weights["to_v_ip.0.weight"])
                attn_procs[name] = ip_processor
        
        self.pipe.unet.set_attn_processor(attn_procs)

        #adapter_modules = torch.nn.ModuleList(self.pipe.unet.attn_processors.values()).to(device)
        attn_processors = self.pipe.unet.attn_processors.values()
        wrapped_processors = [AttnProcessorWrapper(proc) if not isinstance(proc, torch.nn.Module) else proc for proc in attn_processors]
        adapter_modules = torch.nn.ModuleList(wrapped_processors).to(device)
        



        # 步骤2: 创建IPAdapter实例
        self.image_proj_model=image_proj_model
        self.adapter_modules=adapter_modules

        self.pipe.image_proj_model=self.image_proj_model
        self.pipe.adapter_modules=self.adapter_modules

        self.pipe.image_proj_model.requires_grad_(True)
        self.pipe.adapter_modules.requires_grad_(False)
            
        # 日志目录
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)

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
        
        
        #print(self)
        #print("unet.config.cross_attention_dim",self.pipe.unet.config.cross_attention_dim)
        #print("clip_embeddings_dim",self.pipe.image_encoder.config.projection_dim)

        # self.pipe.unet.encoder_hid_proj.image_projection_layers[0].image_embeds = nn.Linear(1024, 4096, device='cuda:0')
        # self.pipe.unet.encoder_hid_proj.image_projection_layers[0].norm = nn.LayerNorm(1024, device='cuda:0')

        # self.pipe.unet.encoder_hid_proj.image_projection_layers.requires_grad_(True)

        self.linear_layer = torch.nn.Linear(2949120, 1024)
        self.linear_layer.requires_grad_(False)


        model = ImageProjModel()
        self.pipe.model = model.to(device)
        self.model = self.pipe.model
        
        # 验证可训练参数数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")  # 应大于0

        print("Modified structure:", self.pipe.unet.encoder_hid_proj)


    
    
    def _preprocess_image(
            self,
            image: Image.Image,
            ref_normal: Optional[Image.Image] = None,
            image_resolution=768,
            mode='stable',
            preprocess='pad'
        ) -> torch.Tensor:

        """图像预处理管道"""
        preprocess='pad'
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        if preprocess == 'pad':
            image, original_size, padding_info = pad_to_square(image)
        elif preprocess == 'crop':
            image, original_size, crop_info = center_crop(image)
        else:
            raise ValueError("Invalid preprocessing mode. Choose 'pad' or 'crop'.")

        image, original_dims, scaling_factors = resize_image(image, image_resolution)

        if ref_normal is not None:
            ref_normal = ref_normal.convert('RGB')
            if preprocess == 'pad':
                ref_normal, _, _ = pad_to_square(ref_normal)
            elif preprocess == 'crop':
                ref_normal, _, _ = center_crop(ref_normal)
            
        # 调整大小
        #ref_normal, _, _ = resize_image(ref_normal, image_resolution)
        
        
        return image.unsqueeze(0).to(self.device)
    
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
        
        # 打印 batch 的类型和内容（前几项）
        # print(f"Batch type: {type(batch)}")
        # print(f"Batch keys: {batch.keys()}")
        # if 'input_images' in batch and len(batch['input_images']) > 0:
        #     first_input_image = batch['input_images']
        #     if isinstance(first_input_image, torch.Tensor):
        #         print(f"First input_image tensor shape: {first_input_image.shape}, dtype: {first_input_image.dtype}")
        #     else:
        #         print(f"First input_image type: {type(first_input_image)}")
        # else:
        #     print("First input_image type: None")

        # 图像预处理
        images = torch.stack([
            self._preprocess_image(img) for img in batch['input_images']
        ]).to(self.device)
        
        
        # print(f"Batch keys: {batch.keys()}")
        # if 'ref_normals' in batch and len(batch['ref_normals']) > 0:
        #     f_ref_normal = batch['ref_normals']
        #     f_ref_normal = f_ref_normal.squeeze(dim=1)
        #     if isinstance(f_ref_normal, torch.Tensor):
        #         print(f"First ref_normals tensor shape: {f_ref_normal.shape}, dtype: {first_input_image.dtype}")
        #         print(f"First ref_normals tensor shape: {f_ref_normal[0].shape}, dtype: {first_input_image.dtype}")
        #     else:
        #         print(f"First ref_normals type: {type(f_ref_normal)}")
        # else:
        #     print("First ref_normals type: None")
        # 参考法线图
        if 'ref_normals' in batch and batch['ref_normals'] is not None:
            ref_normal = torch.stack([
                self._preprocess_image(img) for img in batch['ref_normals']
            ]).to(self.device)
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

        #print("render_gt", _render_gt)
        # render_gt['render_gt'] = _render_gt.to(self.device)


        return lrm_generator_input, render_gt
    
    #试了梯度裁剪 也没有效果
    # def configure_gradient_clipping(
    #     self,
    #     optimizer,
    #     optimizer_idx,
    #     gradient_clip_val=None,
    #     gradient_clip_algorithm=None,
    # ):
    #     # 对所有参数进行梯度裁剪，限制最大范数为1.0
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def forward(self, latents, t, cond_latents, dino_features_ref):
        #gt的latent 循环的次数 起始的条件 嵌入

        #给定起始条件 coarse normal的潜在变量
        cross_attention_kwargs = dict(cond_lat=cond_latents)

        assert torch.isnan(latents).sum() == 0, print(latents)
        assert torch.isnan(t).sum() == 0, print(t)
        assert torch.isnan(dino_features_ref).sum() == 0, print(dino_features_ref)

        # image_embeds = torch.randn(1, 1024, dtype=torch.float32).cuda() * 0.01
        # ip_tokens = model(image_embeds)
        # print("===============FP32 ip_tokens:================", ip_tokens, ip_tokens.max(),ip_tokens.min())

        # 观察给定数据和随机生成的数据取值范围
        #print("-------dino_features_ref-------",dino_features_ref.max(), dino_features_ref.min())
        #dino_randn = torch.randn_like(dino_features_ref)
        #print("-------dino_randn-------", dino_randn.max(), dino_randn.min())
        #dino_randn = dino_randn.float()
        #dino_features_ref = dino_features_ref.float()
        #ip_tokens = self.image_proj_model(dino_randn)
        #ip_tokens = model(dino_features_ref)

        with torch.cuda.amp.autocast():
            ip_tokens = self.pipe.model(dino_features_ref)

        # # ref_normal生成的ip_tokens
        # ip_tokens = self.image_proj_model(dino_features_ref)
        # ip_tokens1 = torch.randn_like(dino_features_ref)
        #print("---32----ip_tokens-------", ip_tokens.max(), ip_tokens.min())
        # ip_tokens1 = ip_tokens1.requires_grad_(True)
        # ip_tokens2 = self.image_proj_model(dino_features_ref)
        ip_tokens = ip_tokens.half()
        #print("----16---ip_tokens-------", ip_tokens.max(), ip_tokens.min())

        encoder_hidden_states = torch.cat([self.prompt_embeds, ip_tokens],dim=1)
        # #encoder_hidden_states = torch.cat([self.prompt_embeds,dino_features_ref], dim = 1)
        # self.check_tensor(ip_tokens, "ip_tokens")
        # self.check_tensor(encoder_hidden_states, "encoder_hidden_states")
        
        prev_noise = self.pipe.dino_unet_forward(
                self.pipe.unet,
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]  # [B,4,h,w]
        # prev_noise = self.pipe.unet(
        #         latents,
        #         t,
        #         encoder_hidden_states,
        #     ).sample


        return prev_noise
    
    def training_step(self, batch, batch_idx):
        # 准备数据
        model_inputs, render_gt = self.prepare_batch_data(batch)
        image = model_inputs['images']
        ref_normal = model_inputs['ref_normals']
        image = image.squeeze(1)  
        ref_normal = ref_normal.squeeze(1)
        render_gt = render_gt.squeeze(1)

        

        # sample random timestep
        B = image.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)


        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            cond_latents = self.encode_condition_image(torch.zeros_like(image))
        else:
            cond_latents = self.encode_condition_image(image)


        train_sched = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.train_scheduler = train_sched
         
   
        latents = self.encode_target_images(render_gt)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
    

        #t_value = t.item()
        ref_normal = ref_normal.to(dtype=next(self.pipe.prior.parameters()).dtype)
        dino_features_ref = self.pipe.prior(ref_normal)
        
        dino_features_ref = dino_features_ref.type(torch.float16)
        dino_features_ref = self.pipe.dino_controlnet.dino_controlnet_cond_embedding(dino_features_ref)
        flattened_tensor = dino_features_ref.reshape(1, -1)

        # 应用线性层
        dino_features_ref = self.linear_layer(flattened_tensor)

        self.check_tensor(latents, "latents")
        self.check_tensor(cond_latents, "cond_latents")
        self.check_tensor(dino_features_ref, "dino_features_ref")

        
        # gt图像的latents 时间步 起始噪声张量 参考图片 目标图像
        prev_noise = self.forward(
                    latents = latents_noisy,
                    t = t,
                    cond_latents=cond_latents,
                    dino_features_ref = dino_features_ref,
                )


        v_target = self.get_v(latents, noise, t).half()


        print("Prev_noise max/min:", prev_noise.max(), prev_noise.min())
        print("v_target max/min:", v_target.max(), v_target.min())
        # 计算损失 预测噪声 & 实际添加噪声
        loss, loss_dict = self.compute_loss(prev_noise, v_target)  
        
        # 日志记录
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        
        self.log('train_loss',
            loss,
            on_step=True, 
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        # 可视化保存（每1000步）
        # if self.global_step % 500 == 0 and self.global_rank == 0:
        #     self._save_training_samples(prev_noise, batch)
        # 释放中间变量
        # del prev_noise, v_target, latents_noisy, noise, cond_latents
        # torch.cuda.empty_cache()  # 可选，但谨慎使用


        return loss.mean()
    

    
    def check_tensor(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN!")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf!")

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

    
    
    
    def _save_training_samples(self, output, batch):
        """保存训练过程样本"""
        # print("`````````````",batch['input_images'].shape)
        # print("`````````````",output.shape)
        output_image = torch.from_numpy(output.prediction).half().permute(0, 3, 1, 2).to(self.device)
        output_image = (output_image + 1) / 2
        grid = torch.cat([
            batch['input_images'][:1], 
            output_image[:1]
        ], dim=-1)
        save_path = os.path.join(self.log_dir, f"*^train_step_{self.global_step}.png")
        save_image(grid, save_path, normalize=True)

    def _save_validation_results(self, output, batch):
        """保存验证结果"""
        val_dir = os.path.join(self.log_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)
        
        for i, (img, normal) in enumerate(zip(batch['input_images'], output['generated_normals'])):
            save_image(normal, os.path.join(val_dir, f"val_{self.current_epoch}_{i}.png"))

    # @torch.no_grad()
    # def encode_condition_image(self, images):
    #     dtype = next(self.pipe.vae.parameters()).dtype
    #     image_pil = [v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])]
    #     image_pt = self.feature_extractor(images=image_pil, return_tensors="pt").pixel_values
    #     image_pt = image_pt.to(device=self.device, dtype=dtype)
    #     latents = self.pipe.vae.encode(image_pt).latent_dist.sample()
    #     return latents
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

        lr = 1e-5

        # 优化器配置
        params = [
            {"params": self.pipe.image_proj_model.parameters()},
            {"params": self.pipe.model.parameters()},
        ]
        
        optimizer = torch.optim.AdamW(params, lr = lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=3000, 
            eta_min=lr/4
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            "gradient_clip_val": 1.0,
        }



    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipe.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8   # [-0.625, 0.625]
        posterior = self.pipe.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents
    

    def get_v(self, x, noise, t):
        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    
    
