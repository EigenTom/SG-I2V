import argparse
import os
import torch
import json
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from einops import rearrange
from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from lvdm.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=int, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    
    # Arguments for Latent Optimization
    parser.add_argument("--use_latent_optimization", action='store_true', help="Enable latent optimization for trajectory control")
    parser.add_argument("--optim_k", type=int, default=10, help="Number of initial steps to apply optimization")
    parser.add_argument("--optim_epochs", type=int, default=5, help="Number of optimization epochs per step")
    parser.add_argument("--optim_lr", type=float, default=0.2, help="Learning rate for latent optimization")
    parser.add_argument("--optim_p_val", type=int, default=100, help="Value of P for top-P selection in loss")
    parser.add_argument("--optim_ref_layers", type=str, nargs='+', 
                        default=['output_blocks.5.1.transformer.transformer_blocks.0.attn2', 
                                 'output_blocks.8.1.transformer.transformer_blocks.0.attn2',
                                 'input_blocks.4.1.transformer.transformer_blocks.0.attn2'],
                        help="List of cross-attention layer names to use for loss")
    parser.add_argument("--optim_bbox_config", type=str, default=None, help="Path to a JSON file with bounding box definitions for each frame")

    return parser

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

def run_inference(args):
    # 1. 加载模型配置和检查点
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda()
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    # 2. 设置采样参数
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    # latent noise shape: 320 // 8 = 40, 512 // 8 = 64 => [40, 64] in spatial dimensions
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    # 3. 加载提示词
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    prompt = prompt_list[0]  # 只使用第一个提示词
    print(f"Using prompt: {prompt}")

    # 4. 准备条件
    text_emb = model.get_learned_conditioning([prompt])
    fps = torch.tensor([args.fps]).to(model.device).long()
    cond = {"c_crossattn": [text_emb], "fps": fps}

    # 5. 设置噪声形状
    noise_shape = [1, channels, frames, h, w]  # batch_size=1

    # Latent Optimization Parameters
    optim_params = None
    if args.use_latent_optimization:
        assert args.optim_bbox_config is not None, "Bounding box config must be provided for latent optimization."
        with open(args.optim_bbox_config, 'r') as f:
            bbox_config = json.load(f)

        optim_params = {
            'k': args.optim_k,
            'epochs': args.optim_epochs,
            'lr': args.optim_lr,
            'P': args.optim_p_val,
            'optim_ref_layers': args.optim_ref_layers,
            'bbox': bbox_config,
        }
        print("Latent optimization enabled with parameters:")
        print(optim_params)

    # 6. 执行DDIM采样
    with torch.no_grad():
        samples = []
        for _ in range(args.n_samples):
            # sample for this prompt
            ddim_sampler = DDIMSampler(model)
            uncond_type = model.uncond_type
            if args.unconditional_guidance_scale != 1.0:
                if uncond_type == "empty_seq":
                    prompts_uc = [""]
                    uc_emb = model.get_learned_conditioning(prompts_uc)
                elif uncond_type == "zero_embed":
                    c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                    uc_emb = torch.zeros_like(c_emb)
                
                if hasattr(model, 'embedder'): # Check for image embedder for unconditional part (though not used in base text-to-video)
                    uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
                    uc_img = model.get_image_embeds(uc_img)
                    uc_emb = torch.cat([uc_emb, uc_img], dim=1)

                if isinstance(cond, dict):
                    uc = {key:cond[key] for key in cond.keys()}
                    uc.update({'c_crossattn': [uc_emb]})
                else:
                    uc = uc_emb
            else:
                uc = None
            
            # Note: x_T is typically None for the first sample unless specific warm-start is intended
            x_T = None 
            
            sample_latent, _ = ddim_sampler.sample(S=args.ddim_steps,
                                                conditioning=cond,
                                                batch_size=noise_shape[0], # Should be 1 for single inference
                                                shape=noise_shape[1:],
                                                verbose=False,
                                                unconditional_guidance_scale=args.unconditional_guidance_scale,
                                                unconditional_conditioning=uc,
                                                eta=args.ddim_eta,
                                                temporal_length=noise_shape[2],
                                                conditional_guidance_scale_temporal=None, # Not in single inference args
                                                x_T=x_T,
                                                use_latent_optimization=args.use_latent_optimization,
                                                optim_params=optim_params
                                                )
            sample_pixel = model.decode_first_stage_2DAE(sample_latent)
            samples.append(sample_pixel)
        
        # 合并所有样本
        if len(samples) > 0:
            samples = torch.cat(samples, dim=0)
        else: # Handle case where n_samples might be 0 or loop doesn't run
            samples = torch.empty(0) # Or handle as an error/default
            
    print(f"[DEBUG]Samples shape: {samples.shape}")

    # batch, num_samples, c, t, h, w
    samples = torch.stack((samples,), dim=1)
    
    # 7. 保存结果
    save_videos(samples, args.savedir, "sample", fps=args.savefps)
    print(f"Saved in {args.savedir}")

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_inference(args)