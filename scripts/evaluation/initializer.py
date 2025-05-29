import torch
from torch import autocast
from einops import rearrange



class initializer:
    def __init__(self, model, sampler, normalize=True):
        self.model = model
        self.sampler = sampler
        self.H = 320
        self.W = 512
        self.T = 16
        self.C = 4
        self.f = 8
        self.shape = [1, self.C, self.T, self.H // self.f, self.W // self.f]    # B, C, T, H, W, set batch size to 1
        self.cond = {'is_use': False}
        
    
    def get_attn(self, prompt, img, scale=1.0, steps=50):
        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for bid, p in enumerate(prompt):
                        uc = self.model.get_learned_conditioning([""])
                        kv = self.model.get_learned_conditioning(p)
                        fps = torch.tensor([8]).to(self.model.device).long()    # 8: fps, parameter
                        c = {"c_crossattn": [kv], "fps": fps}
                        shape = [self.C, self.H // self.f, self.W // self.f]
                        self.sampler.get_attention(S=steps,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=1.0,
                                        x_T=img,
                                        quiet=True,
                                        mask_cond=self.cond,
                                        save_attn_maps=True)
        all_attn_maps = [item[0][0] for item in self.sampler.attn_maps['input_blocks.8.1.transformer_blocks.0.attn2']]
        # avg_maps = [torch.mean(item, axis=0) for item in all_attn_maps]
        # avg_maps = [rearrange(item, 'w h d -> d w h')[None,:] for item in avg_maps]
        # avg_maps = rearrange(torch.cat(avg_maps,dim=0), 't word w h -> word t w h')
        return all_attn_maps