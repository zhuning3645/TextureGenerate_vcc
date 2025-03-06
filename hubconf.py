import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import Optional, Tuple
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from torch.nn.functional import interpolate
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
import torch.nn as nn

dependencies = ["torch", "numpy", "diffusers", "PIL"]

from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
from stablenormal.pipeline_stablenormal import StableNormalPipeline
from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler

def pad_to_square(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int, int, int]]:
    """Pad the input image to make it square."""
    width, height = image.size
    size = max(width, height)
    
    delta_w = size - width
    delta_h = size - height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    padded_image = ImageOps.expand(image, padding)
    return padded_image, image.size, padding

def resize_image(image: Image.Image, resolution: int) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    """Resize the image while maintaining aspect ratio and then pad to nearest multiple of 64."""
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a PIL Image object")
    
    np_image = np.array(image)
    height, width = np_image.shape[:2]

    scale = resolution / min(height, width)
    new_height = int(np.round(height * scale / 64.0)) * 64
    new_width = int(np.round(width * scale / 64.0)) * 64

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image, (height, width), (new_height / height, new_width / width)

def center_crop(image: Image.Image) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float, float, float]]:
    """Crop the center of the image to make it square."""
    width, height = image.size
    crop_size = min(width, height)
    
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image, image.size, (left, top, right, bottom)

class Predictor:
    """Predictor class for Stable Diffusion models."""

    def __init__(self, model):
        self.model = model
        try:
            import xformers
            self.model.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass

    def to(self, device, dtype=torch.float16):
        self.model.to(device, dtype)
        return self
    
    @torch.no_grad()
    def __call__(self, img: Image.Image, ref_normal: Optional[Image.Image] = None, image_resolution=768, mode='stable', preprocess='pad') -> Image.Image:
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        if preprocess == 'pad':
            img, original_size, padding_info = pad_to_square(img)
        elif preprocess == 'crop':
            img, original_size, crop_info = center_crop(img)
        else:
            raise ValueError("Invalid preprocessing mode. Choose 'pad' or 'crop'.")

        img, original_dims, scaling_factors = resize_image(img, image_resolution)

        # 增加对ref_normal的处理逻辑
        if ref_normal is not None:
            ref_normal = ref_normal.convert('RGB')
             # 对 ref_normal 进行相同的预处理
            if preprocess == 'pad':
                ref_normal, _, _ = pad_to_square(ref_normal)
            elif preprocess == 'crop':
                ref_normal, _, _ = center_crop(ref_normal)
            
            # 调整大小
            ref_normal, _, _ = resize_image(ref_normal, 224)
        

        # 在这里结束编辑

        if mode == 'stable':
            init_latents = torch.zeros([1, 4, image_resolution // 8, image_resolution // 8], 
                                    device="cuda", dtype=torch.float16)
        else:
            init_latents = None
        
        print(f"Current model: {self.model.__class__.__name__}")
        print("//////////////////",init_latents)

        
        pipe_out = self.model(
                            img,
                            ref_normal, 
                            match_input_resolution=True, 
                            latents=init_latents,
                            ip_adapter_image_embeds = None
                        )

        # print('pipe_out',pipe_out)
        # 归一化到[0,1]
        pred_normal = (pipe_out.prediction.clip(-1, 1) + 1) / 2
        # 转化为unit8
        pred_normal = (pred_normal[0] * 255).astype(np.uint8)
        # 转换为PIL图像
        pred_normal = Image.fromarray(pred_normal)
        
        new_dims = (int(original_dims[1]), int(original_dims[0])) # reverse the shape (width, height)
        pred_normal = pred_normal.resize(new_dims, Image.Resampling.LANCZOS)

        if preprocess == 'pad':
            left, top, right, bottom = padding_info[0], padding_info[1], original_dims[0] - padding_info[2], original_dims[1] - padding_info[3]
            pred_normal = pred_normal.crop((left, top, right, bottom))
            return pred_normal
        else:
            left, top, right, bottom = crop_info
            pred_normal_with_bg = Image.new("RGB", original_size)
            pred_normal_with_bg.paste(pred_normal, (int(left), int(top)))
            return pred_normal_with_bg
    
    def __repr__(self):
        return f"Predictor(model={self.model})"

def StableNormal(local_cache_dir: Optional[str] = None, device="cuda:0", 
                 yoso_version='yoso-normal-v0-3', diffusion_version='stable-normal-v0-1',
                 ip_adpter_weight: Optional[str] = "ip-adapter_sd15_fixed.safetensors",
                 ref_normal: Optional[Image.Image] = None) -> Predictor:
    """Load the StableNormal pipeline and return a Predictor instance."""
    # 对模型进行初始化
    yoso_weight_path = os.path.join(local_cache_dir if local_cache_dir else "Stable-X", yoso_version)
    diffusion_weight_path = os.path.join(local_cache_dir if local_cache_dir else "Stable-X", diffusion_version)
    
    # yoso_noormal‘s pipeline
    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        yoso_weight_path, trust_remote_code=True, safety_checker=None,
        variant="fp16", torch_dtype=torch.float16).to(device)
    
    # stableNormal's pipeline
    pipe = StableNormalPipeline.from_pretrained(diffusion_weight_path, trust_remote_code=True, safety_checker=None,
                                                variant="fp16", torch_dtype=torch.float16,
                                                scheduler=HEURI_DDIMScheduler(prediction_type='sample', 
                                                                              beta_start=0.00085, beta_end=0.0120, 
                                                                              beta_schedule="scaled_linear"),
                                                low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

    print("process1^^^^^^^^^^^^^^^^^^^^^^^^")
    # 调用方法
    pipe.load_ip_adapter("/data/shared/TextureGeneration/IP-Adapter/IP-Adapter",
                     subfolder="models",
                     weight_name=ip_adpter_weight,
                     local_files_only=True,
                     low_cpu_mem_usage=False)
    
    pipe.x_start_pipeline = x_start_pipeline
    pipe.to(device)
    pipe.prior.to(device, torch.float16)
    
    return Predictor(pipe)

def StableNormal_turbo(local_cache_dir: Optional[str] = None, device="cuda:0", yoso_version='yoso-normal-v0-3') -> Predictor:
    """Load the StableNormal_turbo pipeline for a faster inference."""
    
    yoso_weight_path = os.path.join(local_cache_dir if local_cache_dir else "Stable-X", yoso_version)
    pipe = YOSONormalsPipeline.from_pretrained(yoso_weight_path, 
                                               trust_remote_code=True, safety_checker=None, variant="fp16", 
                                               torch_dtype=torch.float16, t_start=0).to(device)

    return Predictor(pipe)

def _test_run():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image file")
    parser.add_argument("--ref_normal", "-ref", type=str, required=False, help="Ref_detail normal map image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output image file")
    parser.add_argument("--mode", type=str, default="StableNormal_turbo", help="Mode of operation")

    args = parser.parse_args()
    # 1. 加载输入图像
    try:
        image = Image.open(args.input)
        print(f"Loaded input image: {args.input}, size: {image.size}")
    except Exception as e:
        print(f"❌ Failed to load input image: {e}")
        exit(1)
    print("Image path:", args.input)  # 如果从路径加载
    print("Image type:", type(image))  # 检查 image 的类型
    assert image is not None, "Image is None!"

    predictor_func = StableNormal_turbo if args.mode == "StableNormal_turbo" else StableNormal
    predictor = predictor_func(local_cache_dir='./weights', device="cuda:0")
    
    image = Image.open(args.input)
    # 检查 ref_normal 是否为 None
    if args.ref_normal:
        ref_normal = Image.open(args.ref_normal)
        with torch.inference_mode():
            normal_image = predictor(image, ref_normal) 
    else:
        ref_normal = None
        with torch.inference_mode():
            normal_image = predictor(image) 

    
    normal_image.save(args.output)

if __name__ == "__main__":
    _test_run()
