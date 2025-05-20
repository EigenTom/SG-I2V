import os

os.environ["CUDA_VISIBLE_DEVICES"]="0" #restrict CUDA visibility

import cv2
import numpy as np
import random
import math
from PIL import Image
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableVideoDiffusionPipeline, AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel, EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipelineOutput
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims
from IPython.display import HTML

"""
Define utils function
"""

def export_to_gif(
    video_frames, save_path
):
    """
    write to gif
    """
    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    for i in range(len(video_frames)):
        video_frames[i] = Image.fromarray(video_frames[i])
    video_frames[0].save(save_path, save_all=True, append_images=video_frames[1:], loop=0, duration=110)
    return video_frames

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
def tensor2vid(video, processor, output_type: str = "np"):
    #ref: https://github.com/huggingface/diffusers/blob/687bc2772721af584d649129f8d2a28ca56a9ad8/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L61C1-L79C19
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)
    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")    
    return outputs
def visualize_control(image, trajectory_points):
    scale_factor = 1.5
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(trajectory_points)):
        frames = len(trajectory_points[i])
        for j in range(frames):
            source_point = trajectory_points[i][j]
            sx, sy, tx, ty = source_point[0], source_point[1], source_point[2], source_point[3]
            if j==0:
                image2 = cv2.rectangle(image.copy(), (int(sy), int(sx)), (int(ty), int(tx)), (69, 27, 255), thickness=-1)
                image = cv2.rectangle(image, (int(sy), int(sx)), (int(ty), int(tx)), (0, 0, 255), thickness=6)
                image2 = cv2.rectangle(image2, (int(sy), int(sx)), (int(ty), int(tx)), (0, 0, 255), thickness=6)
                image = cv2.addWeighted(image,0.4,image2,0.6,0)
            if j + 1 < frames:
                target_point = trajectory_points[i][j+1]
                sx2, sy2, tx2, ty2 = target_point[0], target_point[1], target_point[2], target_point[3]
                sx3 = (sx+tx)//2
                tx3 = (sx2+tx2)//2
                sy3 = (sy+ty)//2
                ty3 = (sy2+ty2)//2
                arrow_length = np.sqrt((sx3-tx3)**2 + (sy3-ty3)**2)
                green = (0,255,0)
                if j + 2 == frames:
                    image = cv2.line(image, (int(sy3), int(sx3)), (int(ty3), int(tx3)), green, thickness = int(12*scale_factor))
                    image = cv2.circle(image, (int(ty3), int(tx3)), radius = int(15*scale_factor), color = green, thickness = -1)
                    #image = cv2.arrowedLine(image, (int(sy3), int(sx3)), (int(ty3), int(tx3)), green, 12, tipLength=2) #8/arrow_length)
                else:
                    image = cv2.line(image, (int(sy3), int(sx3)), (int(ty3), int(tx3)), green, thickness = int(12*scale_factor)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
def butterworth_low_pass_filter(latents, n=4, d_s=0.25):
    """
    Return the butterworth low pass filter mask.
    Modified from https://github.com/arthur-qiu/FreeTraj/blob/0b3ffdb932bba01ba707d689f4905c31b193468b/utils/utils_freetraj.py#L228
    """
    shape = latents.shape
    H, W = shape[-2], shape[-1]
    mask = torch.zeros_like(latents)
    if d_s==0:
        return mask
    for h in range(H):
        for w in range(W):
            d_square = ((2*h/H-1)**2 + (2*w/W-1)**2)
            mask[..., h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask

# 实现高频噪声替换函数（从 SG-I2V 提取）
def high_frequency_noise_replacement(input_tensor, target_tensor, threshold=0.995):
    """
    使用傅立叶变换实现高频噪声替换
    
    参数:
        input_tensor: 需要被替换高频部分的张量
        target_tensor: 提供高频信息的目标张量
        threshold: 高频区域的阈值，值越高，替换的区域越小
    
    返回:
        融合了目标张量高频信息的输入张量
    """
    # 转换为CUDA张量
    device = input_tensor.device
    
    # 对每个通道分别处理
    result_tensor = torch.zeros_like(input_tensor)
    
    # 遍历所有批次和通道
    for b in range(input_tensor.shape[0]):
        for c in range(input_tensor.shape[1]):
            # 获取当前批次和通道的2D张量
            input_2d = input_tensor[b, c]
            target_2d = target_tensor[b, c]
            
            # 对输入张量进行傅立叶变换
            input_fft = torch.fft.fft2(input_2d)
            input_fft_shifted = torch.fft.fftshift(input_fft)
            
            # 对目标张量进行傅立叶变换
            target_fft = torch.fft.fft2(target_2d)
            target_fft_shifted = torch.fft.fftshift(target_fft)
            
            # 创建频率掩码（中心为低频，边缘为高频）
            h, w = input_2d.shape
            y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            y_coords = y_coords.to(device)
            x_coords = x_coords.to(device)
            
            # 计算到频域中心的距离
            center_y, center_x = h // 2, w // 2
            distance = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            
            # 归一化距离
            max_distance = torch.sqrt(torch.tensor(center_y**2 + center_x**2, device=device))
            normalized_distance = distance / max_distance
            
            # 创建掩码：1表示保留输入，0表示使用目标
            mask = normalized_distance < threshold
            
            # 应用掩码：保留输入的低频，替换为目标的高频
            merged_fft_shifted = torch.where(mask.unsqueeze(-1), input_fft_shifted, target_fft_shifted)
            
            # 反向移位并进行逆傅立叶变换
            merged_fft = torch.fft.ifftshift(merged_fft_shifted)
            merged = torch.fft.ifft2(merged_fft).real
            
            # 存储结果
            result_tensor[b, c] = merged
    
    return result_tensor

"""
Define denoising network (UNet)
"""

class MyUNet(UNetSpatioTemporalConditionModel):
    """
    Modified from SVD implementation
    https://github.com/huggingface/diffusers/blob/24c7d578baf6a8b79890101dd280278fff031d12/src/diffusers/models/unets/unet_spatio_temporal_condition.py#L32
    """
    def inject(self):
        #Replace self-attention blocks in the upsampling layers with our implementation
        for (layer, upsample_block) in enumerate(self.up_blocks):
            if layer == 0: 
                continue
            for (sublayer, trans) in enumerate(upsample_block.attentions):
                basictrans = trans.transformer_blocks[0] #BasicTransformerBlock
                basictrans.attn1.processor = self.my_self_attention(layer, sublayer)

    record_value_ = []
    
    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_time_ids,
        return_dict: bool = True,
    ):
        #Modified from the original implementation such that it cuts redundant computation during the optimization
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = False
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)
        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb
        sample = sample.flatten(0, 1)
        emb = emb.repeat_interleave(num_frames, dim=0)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
        down_block_res_samples = (sample,)
        for layer, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )
        for i, upsample_block in enumerate(self.up_blocks):
            if self.training and i > max(self.record_layer_sublayer)[0]:
                return None #skip redundant computation during optimization
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )
        if self.training:
            return None #skip redundant computation during optimization
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)
    
    def my_self_attention(self, layer, sublayer):
        compress_factor = [None, 4, 2, 1][layer]
        #Modified from the original implementation so that we can record semantically aligned feature maps during the optimization
        def processor(
            attn,
            hidden_states,
            encoder_hidden_states = None,
            attention_mask = None,
            temb = None,
        ):
            residual = hidden_states

            h = self.latent_shape[-2]//compress_factor
            w = self.latent_shape[-1]//compress_factor
            
            input_ndim = hidden_states.ndim
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            if self.training and ((layer, sublayer) in self.record_layer_sublayer):
                #Modified self-attention computation
                #inject key and value from the first frame to obtain semantically aligned fieature maps
                frame = query.shape[0]
                key2 = (key.reshape((1, frame)+key.shape[1:]))[:,:1].repeat((1,frame,1,1,1)).reshape(key.shape)
                value2 = (value.reshape((1, frame)+value.shape[1:]))[:,:1].repeat((1,frame,1,1,1)).reshape(value.shape)

                hidden_states = F.scaled_dot_product_attention(query, key2.clone().detach(), value2.clone().detach(), attn_mask=None, dropout_p=0.0, is_causal=False)
                hid = hidden_states.permute((0, 2, 1, 3)) #(2*batch, h*w, head, channel)
                hid = hid.reshape((hid.shape[0], h, w, -1))
                self.record_value_.append(hid)
            
            hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states
        return processor
    
"""
Define Pipeline
"""

class MyI2VPipe(StableVideoDiffusionPipeline):
    """
    Modified from the original SVD pipeline
    ref: https://github.com/huggingface/diffusers/blob/24c7d578baf6a8b79890101dd280278fff031d12/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L139
    """
    heatmap = {}
    def get_gaussian_heatmap(self, h, w):
        """
        Generate gaussian heatmap
        Modified from https://github.com/showlab/DragAnything/blob/main/demo.py#L380
        """
        if (h,w) in self.heatmap:
            isotropicGrayscaleImage = self.heatmap[(h,w)]
        else:
            sigy = self.unet.heatmap_sigma*(h/2)
            sigx = self.unet.heatmap_sigma*(w/2)

            cx = w/2
            cy = h/2
            isotropicGrayscaleImage = np.zeros((h, w), np.float32)
            for y in range(h):
                for x in range(w):
                    isotropicGrayscaleImage[y, x] = 1 / 2 / np.pi / (sigx*sigy) * np.exp(
                        -1 / 2 * ((x+0.5 - cx) ** 2 / (sigx ** 2) + (y+0.5 - cy) ** 2 / (sigy ** 2)))
            isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
            self.heatmap[(h,w)] = isotropicGrayscaleImage
        return torch.from_numpy(isotropicGrayscaleImage).cuda()
    
    def __call__(self, image, trajectory_points, height, width, num_frames, min_guidance_scale = 1.0, max_guidance_scale = 3.0, fps = 7,
                 generator = None, motion_bucket_id = 127, noise_aug_strength = 0.02, decode_chunk_size = 8):
        #Modified from the original implementaion such that the pipeline incorporates our latent optimization procedure
        
        # set batch size as default to 1
        batch_size = 1
        
        # required by SVD models
        fps = fps - 1
        
        # set the scale of the guidance (VAE?)
        self._guidance_scale = max_guidance_scale
        
        # use CLIP models to encode the image into low-dimension latent features
        image_embeddings = self._encode_image(image, "cuda", 1, self.do_classifier_free_guidance)
        # preprocess the image to specified height and width
        try: 
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
        except:
            self.image_processor = self.video_processor 
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
            
        # add the SLIGHTLY noise to the image (work as DATA AUGMENTATION)
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype).to("cuda")
        print("noise shape: ", noise.shape)
        image = image + noise_aug_strength * noise
        
        # can set the VAE' inference precision to the higher float32
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
            
        # encode the noisy images to the latent space
        image_latents = self._encode_vae_image(
            image,
            device="cuda",
            num_videos_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        
        # enforce the datatype of image latents to be the same as image_embeddings
        image_latents = image_latents.to(image_embeddings.dtype)
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        # expand the image latents to the number of frames
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        # add time_ids, including FPS, motion bucket id, and noise augmentation strength
        # motion_bucket_id control the degree of motion in the generated video
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            1,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to("cuda")
        
        # set the scheduler to the denoising process, specify the timestep of the denoising process
        self.scheduler.set_timesteps(self.unet.num_inference_steps, device="cuda")
        timesteps = self.scheduler.timesteps
        
        # set the number of channels of the latent representations
        num_channels_latents = self.unet.config.in_channels
        
        # define the latent representation's shape
        # corresponding to: B, T, C, H, W
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        
        # generate the initial noisy latent variable with the same random seed
        # the denoising process will be performed from PURE NOISE
        latents = randn_tensor(shape, generator=generator, device=image.device, dtype=image_embeddings.dtype).to("cuda")
        
        # TODO: add proposed noise editing here according to the passed-in trajectory points
        
        
        
        latents = latents * self.scheduler.init_noise_sigma # scale the initial noise by the standard deviation required by the scheduler
        
        # keep record of the shape of latent variable
        self.unet.latent_shape = latents.shape
        
        # create the "guidance scale" for the denoising process: ensure the linear interpolation between the min and max guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to("cuda", latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        # PERF: memory management
        self.vae = self.vae.to("cpu")
        self.image_encoder = self.image_encoder.to("cpu")
        torch.cuda.empty_cache()
            
        #Denoising loop
        
        # set the number of warmup steps
        # warmup steps are the first few steps of the denoising process
        num_warmup_steps = len(timesteps) - self.unet.num_inference_steps * self.scheduler.order #num_warmup_steps = 0 in our setting
        self._num_timesteps = len(timesteps)
        
        
        with self.progress_bar(total=self.unet.num_inference_steps) as progress_bar:
            
            # step through the denoising process recursively
            for i, t in enumerate(timesteps):
                self.unet.cur_timestep = len(timesteps) - i
                
                # if (self.unet.cur_timestep in self.unet.optimize_latent_time):
                #     #update latent through trajectory/object injection and FFT-based high-frequency noise replacement
                
                # setup the current-step random noise which will be added to the latent variable at the current step
                # if classifier-free guidance is used, copy the latents for unconditional and conditional generation
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # perform appropriate scaling of the noise to prepare for passing it to UNet
                # in different timesteps, the noise's strength is scaled differently
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # concatenate the random noise with the encoded image latents as providing conditioning information
                # allowing the UNet to utilize original image's information when predicting the noise
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                
                # generate the predicted noise from the UNet
                with torch.no_grad():   # inference mode
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    
                # classifier-free guidance again
                if self.do_classifier_free_guidance:
                    # split the predicted noise into two parts: unconditional and conditional
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    # final result: linear interpolation between the unconditional and conditional noise
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # key step: scheduler compute the next timestep's latent variable
                # according to the predicted noise, the current latent variable, and the current timestep
                # will subtract the predicted noise from the current latent variable by some scaling factor determined by the scheduler
                # might add some small perbutation noise, if s_churn is not 0
                latents = self.scheduler.step(noise_pred, t, latents,  s_churn = 0.0).prev_sample
                
                # purely visualization purpose
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    
        # prepare for the decoding process            
        self.vae = self.vae.to("cuda")
        self.image_encoder = self.image_encoder.to("cuda")
        
        # set the precision of the VAE to the original float16
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        # decode the latent variable to the image space
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type="np")
        
        self.maybe_free_model_hooks()
        return StableVideoDiffusionPipelineOutput(frames=frames)
    
class Config:
    """
    Hyperparameters
    """
    seed = 817
    height, width = 576, 1024 #resolution of generated video
    num_frames = 14
    num_inference_steps = 50 #total number of inference steps
    optimize_latent_time = list(range(30,46)) #set of timesteps to perform optimization
    optimize_latent_iter = 5 #number of optimization iterations to perform for each timestep
    optimize_latent_lr = 0.21 #learning rate for optimization
    record_layer_sublayer = [(2, 1), (2, 2)] #extract feature maps from 1st and 2nd self-attention (note: 0-indexed base) located at 2nd resolution-level of upsampling layer
    heatmap_sigma = 0.4 #standard deviation of gaussian heatmap
    fft_ratio = 0.5 #fft mix ratio
    latent_fft_post_merge = True #fft-based post-processing is enabled iff True

config = Config()

# Load and set up the pipeline
svd_dir = "stabilityai/stable-video-diffusion-img2vid"
cache_dir = "./../"
    
feature_extractor = CLIPImageProcessor.from_pretrained(svd_dir, subfolder="feature_extractor", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_dir, subfolder="vae", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16").to("cuda")
requires_grad(vae, False)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(svd_dir, subfolder="image_encoder", cache_dir = cache_dir, torch_dtype=torch.float16, variant="fp16").to("cuda")
requires_grad(image_encoder, False)
unet = MyUNet.from_pretrained(svd_dir, subfolder="unet", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16").to("cuda")
requires_grad(unet, False)
scheduler = EulerDiscreteScheduler.from_pretrained(svd_dir, subfolder="scheduler", cache_dir=cache_dir, torch_dtype=torch.float16, variant="fp16")
        
# unet.inject() #inject module
    
#Set up pipeline
pipe = MyI2VPipe(vae,image_encoder,unet,scheduler,feature_extractor).to(device="cuda")

#Input example 1
"""
bounding_box : N x (w_1, h_1, w_2, h_2), specifying corner coordinates of bounding boxes on the first frame
center_traj: N x F x 2, specifying center coordinates (w, h) of bounding boxes at each frame (at the scale of config.width and config.height)
"""
image_url = "./examples/111/img.png"
bounding_box = [[290.,   4., 541.,  74.]] 
center_traj = [
       [[415.        ,  39.        ],
        [430. ,  34.66408739],
        [445.,  31.64997724],
        [460.,  29.71734183],
        [475.,  28.62585344],
        [490.,  28.13518434],
        [505.,  28.00500683],
        [520.,  28.2553482 ],
        [535.,  30.20801092],
        [550.,  33.8830223 ],
        [565.,  39.04005462],
        [565.,  39.04005462],
        [565.,  39.04005462],
        [565.,  39.04005462],
        ]]
assert(len(bounding_box)==len(center_traj))

#load image
image = Image.open(image_url).convert('RGB')
image = image.resize((config.width, config.height))

#preprocess trajectory
bounding_box = np.array(bounding_box).astype(np.float32)
center_traj = np.array(center_traj).astype(np.float32)
trajectory_points = [] # N x frames x 4
for j, trajectory in enumerate(center_traj):
        #For normal use
        box_traj = [] # frames x 4
        for i in range(config.num_frames):
            d = center_traj[j][i] - center_traj[j][0]
            dx, dy = d[0], d[1]
            box_traj.append(np.array([bounding_box[j][1] + dy, bounding_box[j][0] + dx, bounding_box[j][3] + dy, bounding_box[j][2] + dx], dtype=np.float32))
        trajectory_points.append(box_traj)

#Approx. 4 minutes on A6000 with default config
def run(config, image, trajectory_points, save_path):
    """
    Set up hyperparameters
    """
    pipe.unet.num_inference_steps = config.num_inference_steps
    pipe.unet.optimize_zero_initialize_param = True
    height, width = config.height, config.width
    motion_bucket_id = 127
    fps = 7
    num_frames = config.num_frames
    seed = config.seed
    pipe.unet.heatmap_sigma = config.heatmap_sigma
    pipe.unet.latent_fft_post_merge = config.latent_fft_post_merge
    pipe.unet.latent_fft_ratio = config.fft_ratio #range : 0.0 - 1.0
    pipe.unet.optimize_latent_iter = config.optimize_latent_iter
    pipe.unet.optimize_latent_lr = config.optimize_latent_lr
    pipe.unet.optimize_latent_time = config.optimize_latent_time
    pipe.unet.record_layer_sublayer =  config.record_layer_sublayer
    generator = torch.manual_seed(seed)
    frames = pipe(image, trajectory_points, height=height, width=width, num_frames = num_frames, decode_chunk_size=8, generator=generator, fps=fps, motion_bucket_id=motion_bucket_id, noise_aug_strength=0.02).frames[0]
    #save video
    export_to_gif(frames, save_path)

#generate video
run(config, image, trajectory_points, save_path="./result.gif")
HTML("<img src=\"" + "./result.gif" + "\">")