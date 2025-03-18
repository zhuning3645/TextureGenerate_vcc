# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# --------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
from safetensors import safe_open
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.loaders import IPAdapterMixin
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, UNet2DConditionModel

from diffusers.image_processor import PipelineImageInput
from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
)
from diffusers.schedulers import (
    DDIMScheduler
)

from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers



from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline, MultiControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0 as IPAttnProcessor
from src.attention_processor import CNAttnProcessor2_0 as CNAttnProcessor


import torch.nn as nn
import torch.nn.functional as F





logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
Examples:
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
...     "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> normals = pipe(image)

>>> vis = pipe.image_processor.visualize_normals(normals.prediction)
>>> vis[0].save("einstein_normals.png")
```
"""


@dataclass
class StableNormalOutput(BaseOutput):
    """
    Output class for Marigold monocular normals prediction pipeline.

    Args:
        prediction (`np.ndarray`, `torch.Tensor`):
            Predicted normals with values in the range [-1, 1]. The shape is always $numimages \times 3 \times height
            \times width$, regardless of whether the images were passed as a 4D array or a list.
        uncertainty (`None`, `np.ndarray`, `torch.Tensor`):
            Uncertainty maps computed from the ensemble, with values in the range [0, 1]. The shape is $numimages
            \times 1 \times height \times width$.
        latent (`None`, `torch.Tensor`):
            Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline.
            The shape is $numimages * numensemble \times 4 \times latentheight \times latentwidth$.
    """

    prediction: Union[np.ndarray, torch.Tensor]
    latent: Union[None, torch.Tensor]
    gaus_noise: Union[None, torch.Tensor]


from einops import rearrange  
class DINOv2_Encoder(torch.nn.Module):
    IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name = 'dinov2_vitl14',
        freeze = True,
        antialias=True,
        device="cuda",
        size = 448,
    ):
        super(DINOv2_Encoder, self).__init__()
        
        self.model = torch.hub.load('./dinov2', model_name, source='local')
        self.model.eval().to(device)
        self.device = device
        self.antialias = antialias
        self.dtype = torch.float32

        self.mean = torch.Tensor(self.IMAGENET_DEFAULT_MEAN)
        self.std = torch.Tensor(self.IMAGENET_DEFAULT_STD)
        self.size = size
        if freeze:
            self.freeze()


    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encoder(self, x):
        '''
        x: [b h w c], range from (-1, 1), rbg
        '''

        x = self.preprocess(x).to(self.device, self.dtype)

        b, c, h, w = x.shape
        patch_h, patch_w = h // 14, w // 14

        embeddings = self.model.forward_features(x)['x_norm_patchtokens']
        embeddings = rearrange(embeddings, 'b (h w) c -> b h w c', h = patch_h, w = patch_w)

        return  rearrange(embeddings, 'b h w c -> b c h w')

    def preprocess(self, x):
        ''' x
        '''
        # normalize to [0,1],
        x = torch.nn.functional.interpolate(
            x,
            size=(self.size, self.size),
            mode='bicubic',
            align_corners=True,
            antialias=self.antialias,
        )

        x = (x + 1.0) / 2.0
        # renormalize according to dino
        mean = self.mean.view(1, 3, 1, 1).to(x.device)
        std = self.std.view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        return x
    
    def to(self, device, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.model.to(device, dtype)
            self.mean.to(device, dtype)
            self.std.to(device, dtype)
        else:
            self.model.to(device)
            self.mean.to(device)
            self.std.to(device)
        return self

    def __call__(self, x, **kwargs):
        return self.encoder(x, **kwargs)

class StableNormalPipeline(StableDiffusionControlNetPipeline, IPAdapterMixin):
    """ Pipeline for monocular normals estimation using the Marigold method: https://marigoldmonodepth.github.io.
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]



    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        dino_controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        scheduler: Union[DDIMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
        default_denoising_steps: Optional[int] = 10,
        default_processing_resolution: Optional[int] = 768,
        prompt="The normal map",
        empty_text_embedding=None,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
                )

        self.register_modules(
            dino_controlnet=dino_controlnet,
        )

        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.dino_image_processor = lambda x: x / 127.5 -1.

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.prompt = prompt
        self.prompt_embeds = None
        self.empty_text_embedding = empty_text_embedding
        self.prior = DINOv2_Encoder(size=672)
        self.clip_image_processor = CLIPImageProcessor()


        self.num_tokens = 4
        # self.image_encoder.config.projection_dim = 1024

        self.set_ip_adapter()
        self.image_proj_model = self.init_proj().to(self.device)
        df_model = DinoFeatureModel().type(torch.float16)
        self.df_model = df_model.to("cuda:0")

      
    def check_inputs(
        self,
        image: PipelineImageInput,
        ref_normal: PipelineImageInput,
        num_inference_steps: int,
        ensemble_size: int,
        processing_resolution: int,
        resample_method_input: str,
        resample_method_output: str,
        batch_size: int,
        ensembling_kwargs: Optional[Dict[str, Any]],
        latents: Optional[torch.Tensor],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        output_type: str,
        output_uncertainty: bool,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
    ) -> int:
        if num_inference_steps is None:
            raise ValueError("`num_inference_steps` is not specified and could not be resolved from the model config.")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be positive.")
        if ensemble_size < 1:
            raise ValueError("`ensemble_size` must be positive.")
        if ensemble_size == 2:
            logger.warning(
                "`ensemble_size` == 2 results are similar to no ensembling (1); "
                "consider increasing the value to at least 3."
            )
        if ensemble_size == 1 and output_uncertainty:
            raise ValueError(
                "Computing uncertainty by setting `output_uncertainty=True` also requires setting `ensemble_size` "
                "greater than 1."
            )
        if processing_resolution is None:
            raise ValueError(
                "`processing_resolution` is not specified and could not be resolved from the model config."
            )
        if processing_resolution < 0:
            raise ValueError(
                "`processing_resolution` must be non-negative: 0 for native resolution, or any positive value for "
                "downsampled processing."
            )
        if processing_resolution % self.vae_scale_factor != 0:
            raise ValueError(f"`processing_resolution` must be a multiple of {self.vae_scale_factor}.")
        if resample_method_input not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_input` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if resample_method_output not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_output` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        if output_type not in ["pt", "np"]:
            raise ValueError("`output_type` must be one of `pt` or `np`.")
        if latents is not None and generator is not None:
            raise ValueError("`latents` and `generator` cannot be used together.")
        if ensembling_kwargs is not None:
            if not isinstance(ensembling_kwargs, dict):
                raise ValueError("`ensembling_kwargs` must be a dictionary.")
            if "reduction" in ensembling_kwargs and ensembling_kwargs["reduction"] not in ("closest", "mean"):
                raise ValueError("`ensembling_kwargs['reduction']` can be either `'closest'` or `'mean'`.")

        # image checks
        num_images = 0
        W, H = None, None
        if not isinstance(image, list):
            image = [image]
        for i, img in enumerate(image):
            if isinstance(img, np.ndarray) or torch.is_tensor(img):
                if img.ndim not in (2, 3, 4):
                    raise ValueError(f"`image[{i}]` has unsupported dimensions or shape: {img.shape}.")
                H_i, W_i = img.shape[-2:]
                N_i = 1
                if img.ndim == 4:
                    N_i = img.shape[0]
            elif isinstance(img, Image.Image):
                W_i, H_i = img.size
                N_i = 1
            else:
                raise ValueError(f"Unsupported `image[{i}]` type: {type(img)}.")
            if W is None:
                W, H = W_i, H_i
            elif (W, H) != (W_i, H_i):
                raise ValueError(
                    f"Input `image[{i}]` has incompatible dimensions {(W_i, H_i)} with the previous images {(W, H)}"
                )
            num_images += N_i

        # latents checks
        if latents is not None:
            if not torch.is_tensor(latents):
                raise ValueError("`latents` must be a torch.Tensor.")
            if latents.dim() != 4:
                raise ValueError(f"`latents` has unsupported dimensions or shape: {latents.shape}.")

            if processing_resolution > 0:
                max_orig = max(H, W)
                new_H = H * processing_resolution // max_orig
                new_W = W * processing_resolution // max_orig
                if new_H == 0 or new_W == 0:
                    raise ValueError(f"Extreme aspect ratio of the input image: [{W} x {H}]")
                W, H = new_W, new_H
            w = (W + self.vae_scale_factor - 1) // self.vae_scale_factor
            h = (H + self.vae_scale_factor - 1) // self.vae_scale_factor
            shape_expected = (num_images * ensemble_size, self.vae.config.latent_channels, h, w)

            if latents.shape != shape_expected:
                raise ValueError(f"`latents` has unexpected shape={latents.shape} expected={shape_expected}.")

        # generator checks
        if generator is not None:
            if isinstance(generator, list):
                if len(generator) != num_images * ensemble_size:
                    raise ValueError(
                        "The number of generators must match the total number of ensemble members for all input images."
                    )
                if not all(g.device.type == generator[0].device.type for g in generator):
                    raise ValueError("`generator` device placement is not consistent in the list.")
            elif not isinstance(generator, torch.Generator):
                raise ValueError(f"Unsupported generator type: {type(generator)}.")
            
        # ip-adapter checks
        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

        return num_images

    def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(**self._progress_bar_config)
        progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
        progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim = self.unet.config.cross_attention_dim,
            clip_embed_dim = 1024,
            num_tokens = self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self, "controlnet"):
            if isinstance(self.controlnet, MultiControlNetModel):
                for controlnet in self.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))


    def load_ip_adapter(self, ckpt_path: str):
        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"image_proj_model": {}, "adapter_modules": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj_model."):
                        state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                    elif key.startswith("adapter_modules."):
                        state_dict["adapter_modules"][key.replace("adapter_modules.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        
        # state_dict = torch.load("/data/shared/TextureGeneration/Texture/logs/TextureGenera-train-exp86/checkpoints/step=00002000.ckpt")
        model_weights = state_dict["state_dict"]
        # print(model_weights.keys())
        
        image_proj_weights = {
            k.replace("image_proj_model.", "", 1): v 
            for k, v in model_weights.items() 
            if k.startswith("image_proj_model.")
        }
        self.image_proj_model.load_state_dict(image_proj_weights)

        # ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        attn_processors = self.unet.attn_processors.values()
        wrapped_processors = [AttnProcessorWrapper(proc) if not isinstance(proc, torch.nn.Module) else proc for proc in attn_processors]
        ip_layers = torch.nn.ModuleList(wrapped_processors)
        adapter_modules_weights = {
            k.replace("adapter_modules.", "", 1): v 
            for k, v in model_weights.items() 
            if k.startswith("adapter_modules.")
        }
        ip_layers.load_state_dict(adapter_modules_weights)

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            # 这里改成使用dino作为image的解码器
            # clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            dino_features_ref = self.prior(clip_image.to(self.device, dtype=torch.float16))
            dino_features_ref = self.dino_controlnet.dino_controlnet_cond_embedding(dino_features_ref)
            dino_features_ref = self.df_model(dino_features_ref).to(self.device)
        else:
            dino_features_ref = clip_image_embeds.to(self.device, dtype=torch.float16)


        self.image_proj_model = self.image_proj_model.to(self.device)
        image_prompt_embeds = self.image_proj_model(dino_features_ref)

        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(dino_features_ref))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def encode_condition_image(self, images):
        dtype = next(self.vae.parameters()).dtype
        
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
        latents = self.vae.encode(normalized.to(dtype)).latent_dist.sample()
        return latents
    
    '''
    ip_adapter_image: 
        输入的图像数据,可以是单个图像或者多个图像组成的列表,每个图像用于一个独立的IP Adapter.
    ip_adapter_image_embeds: 
        预先计算好的图像嵌入，如果提供了这个参数，则直接使用这些嵌入，而不从头计算。
    device: 
        GPU&CPU
    num_images_per_prompt: 
        每个提示(prompt)生成的图像数量。
    do_classifier_free_guidance: 
        是否执行分类器自由引导(classifier-free guidance)
    '''



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        ref_normal: Optional[Image.Image] = None,  # 新增参数
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        output_type: str = "np",
        output_uncertainty: bool = False,
        output_latent: bool = False,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`),
                `List[torch.Tensor]`: An input image or images used as an input for the normals estimation task. For
                arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible
                by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or
                three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the
                same width and height.
            num_inference_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, defaults to `1`):
                Number of ensemble predictions. Recommended values are 5 and higher for better precision, or 1 for
                faster inference.
            processing_resolution (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters when setting `ensemble_size` or passing a tensor of images.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"closest"`): Defines the ensembling function applied in
                  every pixel location, can be either `"closest"` or `"mean"`.
            latents (`torch.Tensor`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_type (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `latents` argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.marigold.MarigoldDepthOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.marigold.MarigoldNormalsOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.marigold.MarigoldNormalsOutput`] is returned, otherwise a
                `tuple` is returned where the first element is the prediction, the second element is the uncertainty
                (or `None`), and the third is the latent (or `None`).
        """
        
        # 0. Resolving variables.
        device = self._execution_device
        dtype = self.dtype

        # Model-specific optimal default values leading to fast and reasonable results.
        if num_inference_steps is None:
            num_inference_steps = self.default_denoising_steps
        if processing_resolution is None:
            processing_resolution = self.default_processing_resolution


        image, padding, original_resolution = self.image_processor.preprocess(
            image, processing_resolution, resample_method_input, device, dtype
        )  # [N,3,PPH,PPW]

        image_latent, gaus_noise = self.prepare_latents(
            image, latents, generator, ensemble_size, batch_size
        )  # [N,4,h,w], [N,4,h,w]


        predictor = self.x_start_pipeline(image, latents=gaus_noise, 
                                    processing_resolution=processing_resolution, skip_preprocess=True)
        x_start_latent = predictor.latent

        # 1. Check inputs.
        num_images = self.check_inputs(
            image,
            ref_normal,
            num_inference_steps,
            ensemble_size,
            processing_resolution,
            resample_method_input,
            resample_method_output,
            batch_size,
            ensembling_kwargs,
            latents,
            generator,
            output_type,
            output_uncertainty,
        )

        # 2. Prepare empty text conditioning.
        # Model invocation: self.tokenizer, self.text_encoder.
        if self.empty_text_embedding is None:
            prompt = ""
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            self.empty_text_embedding = self.text_encoder(text_input_ids)[0]  # [1,2,1024]


        # 3. prepare prompt
        if self.prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                self.prompt,
                device,
                num_images_per_prompt,
                False,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            self.prompt_embeds = prompt_embeds
            self.negative_prompt_embeds = negative_prompt_embeds


        # 5. dino guider features obtaining
        ## TODO different case-1
        dino_features = self.prior(image)
        dino_features = self.dino_controlnet.dino_controlnet_cond_embedding(dino_features)
        # dino_features = self.match_noisy(dino_features, x_start_latent)

        del (
                image,
        )

        '''
        ref_normal的预处理
        '''
        num_samples = 4

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=ref_normal, clip_image_embeds=None
        )# 1 4 1024

        # bs_embed, seq_len, _ = image_prompt_embeds.shape
        # image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        # image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        # uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        # bs 4 1024 + bs 77 1024 = bs 81 1024
        encoder_hidden_states = torch.cat([self.prompt_embeds, image_prompt_embeds], dim = 1)

        # 7. denoise sampling, using heuritic sampling proposed by Ye.

        t_start = self.x_start_pipeline.t_start
        self.scheduler.set_timesteps(num_inference_steps, t_start=t_start,device=device)

        cond_scale = controlnet_conditioning_scale
        pred_latent = image_latent
        # 这里直接使用image_latent作为cond_latent, 原版是x_start_latent

        cur_step = 0

        # # dino controlnet
        # dino_down_block_res_samples, dino_mid_block_res_sample = self.dino_controlnet(
        #     dino_features.detach(),
        #     0, # not depend on time steps
        #     encoder_hidden_states=self.prompt_embeds,
        #     conditioning_scale=cond_scale,
        #     guess_mode=False,
        #     return_dict=False,
        # )
        # assert dino_mid_block_res_sample == None

        pred_latents = []

        # Denoising loop
        last_pred_latent = pred_latent
        for (t, prev_t) in self.progress_bar(zip(self.scheduler.timesteps,self.scheduler.prev_timesteps), leave=False, desc="Diffusion steps..."):

            #_dino_down_block_res_samples = [dino_down_block_res_sample for dino_down_block_res_sample in dino_down_block_res_samples]  # copy, avoid repeat quiery

            # controlnet
            # down_block_res_samples, mid_block_res_sample = self.controlnet(
            #     image_latent.detach(),
            #     t,
            #     encoder_hidden_states=self.prompt_embeds,
            #     conditioning_scale=cond_scale,
            #     #controlnet_cond = [image_latent, image_latent],
            #     guess_mode=False,
            #     return_dict=False,
            # )

            # SG-DRN
            # 在每个时间步t中，模型使用DINO UNet进行去噪操作 
            noise = self.dino_unet_forward(
                self.unet,
                pred_latent,
                t,
                encoder_hidden_states = encoder_hidden_states,
                # cross_attention_kwargs=self.cross_attention_kwargs,
                #dino_down_block_additional_residuals=_dino_down_block_res_samples,
                return_dict=False,
            )[0]  # [B,4,h,w]


            pred_latents.append(noise)
            # ddim steps 利用调度器更新潜在变量
            out = self.scheduler.step(
                noise, t, prev_t, pred_latent, gaus_noise = gaus_noise, generator=generator, cur_step=cur_step+1  # NOTE that cur_step dirs to next_step
            )# [B,4,h,w]
            pred_latent = out.prev_sample

            cur_step += 1

        del (
            image_latent,
            dino_features,
        )
        pred_latent = pred_latents[-1]  # using x0

        # decoder
        prediction = self.decode_prediction(pred_latent)
        prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E,3,PH,PW]
        prediction = self.image_processor.resize_antialias(prediction, original_resolution, resample_method_output, is_aa=False)  # [N,3,H,W]

        if match_input_resolution:
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [N,3,H,W]

        if match_input_resolution:
            prediction = self.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [N,3,H,W]
        prediction = self.normalize_normals(prediction)  # [N,3,H,W]

        if output_type == "np":
            prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,3]
            prediction = prediction.clip(min=-1, max=1)

        # 11. Offload all models
        self.maybe_free_model_hooks()

        return StableNormalOutput(
            prediction=prediction,
            latent=pred_latent,
            gaus_noise=gaus_noise
        )

    # Copied from diffusers.pipelines.marigold.pipeline_marigold_depth.MarigoldDepthPipeline.prepare_latents
    def prepare_latents(
        self,
        image: torch.Tensor,
        latents: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def retrieve_latents(encoder_output):
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")



        image_latent = torch.cat(
            [
                retrieve_latents(self.vae.encode(image[i : i + batch_size]))
                for i in range(0, image.shape[0], batch_size)
            ],
            dim=0,
        )  # [N,4,h,w]
        image_latent = image_latent * self.vae.config.scaling_factor
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]

        pred_latent = latents
        if pred_latent is None:


            pred_latent = randn_tensor(
                image_latent.shape,
                generator=generator,
                device=image_latent.device,
                dtype=image_latent.dtype,
            )  # [N*E,4,h,w]

        return image_latent, pred_latent

    def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:
        if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:
            raise ValueError(
                f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
            )

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]

        return prediction  # [B,3,H,W]

    @staticmethod
    def normalize_normals(normals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if normals.dim() != 4 or normals.shape[1] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")

        norm = torch.norm(normals, dim=1, keepdim=True)
        normals /= norm.clamp(min=eps)

        return normals

    @staticmethod
    def match_noisy(dino, noisy):
        _, __, dino_h, dino_w =  dino.shape
        _, __, h, w =  noisy.shape

        if h == dino_h and w == dino_w:
            return dino
        else:
            return F.interpolate(dino, (h, w), mode='bilinear')


    
    







    @staticmethod
    def dino_unet_forward(
        self,  # NOTE that repurpose to UNet
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        dino_down_block_additional_residuals: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.


        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True



        def residual_downforward(
            self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None,
            additional_residuals: Optional[torch.Tensor] = None,
            *args, **kwargs,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                deprecate("scale", "1.0.0", deprecation_message)

            output_states = ()

            for resnet in self.resnets:
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb)
                    #hidden_states += additional_residuals.pop(0)


                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
                    #hidden_states += additional_residuals.pop(0)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states


        def residual_blockforward(
            self,  ## NOTE that repurpose to unet_blocks
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            additional_residuals: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")



            output_states = ()

            blocks = list(zip(self.resnets, self.attentions))

            for i, (resnet, attn) in enumerate(blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                #hidden_states += additional_residuals.pop(0)

                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
                    #hidden_states += additional_residuals.pop(0)

                output_states = output_states + (hidden_states,)

            return hidden_states, output_states


        down_intrablock_additional_residuals = dino_down_block_additional_residuals

        #sample += down_intrablock_additional_residuals.pop(0)
        down_block_res_samples = (sample,)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:

                sample, res_samples = residual_blockforward(
                    downsample_block,
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    additional_residuals = down_intrablock_additional_residuals,
                )

            else:
                sample, res_samples = residual_downforward(
                    downsample_block,
                    hidden_states=sample,
                    temb=emb,
                    additional_residuals = down_intrablock_additional_residuals,
                        )


            down_block_res_samples += res_samples


        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)



    @staticmethod
    def ensemble_normals(
        normals: torch.Tensor, output_uncertainty: bool, reduction: str = "closest"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Ensembles the normals maps represented by the `normals` tensor with expected shape `(B, 3, H, W)`, where B is
        the number of ensemble members for a given prediction of size `(H x W)`.

        Args:
            normals (`torch.Tensor`):
                Input ensemble normals maps.
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                Whether to output uncertainty map.
            reduction (`str`, *optional*, defaults to `"closest"`):
                Reduction method used to ensemble aligned predictions. The accepted values are: `"closest"` and
                `"mean"`.

        Returns:
            A tensor of aligned and ensembled normals maps with shape `(1, 3, H, W)` and optionally a tensor of
            uncertainties of shape `(1, 1, H, W)`.
        """
        if normals.dim() != 4 or normals.shape[1] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")
        if reduction not in ("closest", "mean"):
            raise ValueError(f"Unrecognized reduction method: {reduction}.")

        mean_normals = normals.mean(dim=0, keepdim=True)  # [1,3,H,W]
        mean_normals = MarigoldNormalsPipeline.normalize_normals(mean_normals)  # [1,3,H,W]

        sim_cos = (mean_normals * normals).sum(dim=1, keepdim=True)  # [E,1,H,W]
        sim_cos = sim_cos.clamp(-1, 1)  # required to avoid NaN in uncertainty with fp16

        uncertainty = None
        if output_uncertainty:
            uncertainty = sim_cos.arccos()  # [E,1,H,W]
            uncertainty = uncertainty.mean(dim=0, keepdim=True) / np.pi  # [1,1,H,W]

        if reduction == "mean":
            return mean_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

        closest_indices = sim_cos.argmax(dim=0, keepdim=True)  # [1,1,H,W]
        closest_indices = closest_indices.repeat(1, 3, 1, 1)  # [1,3,H,W]
        closest_normals = torch.gather(normals, 0, closest_indices)  # [1,3,H,W]

        return closest_normals, uncertainty  # [1,3,H,W], [1,1,H,W]

class ImageProjModel(torch.nn.Module):
    def __init__(self, clip_embed_dim=1024, cross_attention_dim=1024, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Linear(clip_embed_dim, cross_attention_dim * num_tokens)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        # image_embeds: [B, clip_embed_dim]
        x = self.proj(image_embeds)  # [B, cross_attention_dim * num_tokens]
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)  # [B, num_tokens, cross_attention_dim]
        x = self.norm(x)
        return x

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class DinoFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Linear(320, 1024)                   # 线性映射

    def forward(self, x):
        x = self.global_pool(x)  # [batch, 320, 1, 1]
        x = x.flatten(1)         # [batch, 320]
        x = self.fc(x)            # [batch, 1024]
        return x

# 图像投影模型：将CLIP图像特征映射为提示词序列
class ImageProjModel(torch.nn.Module):
    def __init__(self, clip_embed_dim=1024, cross_attention_dim=1024, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.generator = None

        self.proj = torch.nn.Linear(clip_embed_dim, self.num_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        
    def forward(self, image_embeds):
        # image_embeds: [B, clip_embed_dim]

        assert not torch.isnan(image_embeds).any(), "输入包含NaN"

        
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds)
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(
            -1, self.num_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
    
        return clip_extra_context_tokens
    
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
                        


class AttnProcessorWrapper(torch.nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def forward(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

