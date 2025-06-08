import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

class LatentOptimizer:
    def __init__(self, model, optim_params):
        self.model = model
        self.optim_params = optim_params

    def get_attention_maps(self, layer_names):
        """
        Extracts attention maps from specified layers of the model.
        """
        attn_maps = {}
        for name, module in self.model.model.diffusion_model.named_modules():
            module_name = type(module).__name__
            if module_name == 'CrossAttention' and 'attn2' in name:
                if hasattr(module, 'attn_maps') and module.attn_maps is not None and name in layer_names:
                    # The attention map is saved in the module during the forward pass.
                    # In `lvdm.modules.attention.CrossAttention`, `self.attn_map` is stored.
                    # Its shape is (batch, heads, sequence_length_q, sequence_length_k).
                    # For our case, this is (b*f, num_heads, h*w, num_tokens).
                    # print(f"[DEBUG] checked name: {name}")
                    # print(f"[DEBUG] check module: {module.attn_maps.shape}")
                    attn_maps[name] = module.attn_maps
        return attn_maps

    def _get_layer_resolution(self, layer_name):
        # This function infers the spatial resolution of the feature map at a given layer.
        # It's based on the U-Net architecture where resolution is halved at certain stages.
        # `attention_resolutions` in the config are [4, 2, 1].
        # These correspond to `ds` values which are powers of 2, representing downsampling factor.
        # ds=1 -> 1/8 of latent size, ds=2 -> 1/16, ds=4 -> 1/32
        # Base latent size (h,w) is (40, 64) for 320x512 input.
        
        # Let's map block names to downsampling factors (ds)
        # The UNet has input_blocks, a middle_block, and output_blocks.
        # `channel_mult` is (1, 2, 4, 4), len=4, so 4 levels.
        # Level 0: no downsample (ds=1)
        # Level 1: ds=2
        # Level 2: ds=4
        # Level 3: ds=8
        # The attention resolutions are checked against `ds`.
        # So we have attention at ds=4 and ds=2 and ds=1... wait, this is confusing.
        
        # Simpler approach: let's trace layer names.
        # input_blocks are indexed 0 to 11.
        # middle_block is one block.
        # output_blocks are indexed 0 to 11.
        
        # From my previous analysis:
        # `input_blocks.8.1` -> this is deep in the encoder
        # `output_blocks.3.1`, `output_blocks.4.1`, `output_blocks.5.1` -> decoder
        
        # The feature map size halves at blocks 1, 2, 4 in input_blocks
        # and doubles at blocks 7, 8, 10 in output_blocks (roughly).
        
        # Based on `openaimodel3d.py`, the `ds` value increases in the input blocks
        # and decreases in the output blocks.
        # Let's create a rough mapping.
        # initial h, w
        # will be 40, 64
        h, w = self.model.image_size[0], self.model.image_size[1]
        
        if 'input_blocks.4' in layer_name or 'output_blocks.7' in layer_name: # 2x downsample
            return h // 2, w // 2 # 20, 32#
        if 'input_blocks.5' in layer_name or 'output_blocks.6' in layer_name: # 2x downsample
            return h // 2, w // 2 # 20, 32
        if 'input_blocks.7' in layer_name or 'output_blocks.4' in layer_name: # 4x downsample
            return h // 4, w // 4 # 10, 16
        if 'input_blocks.8' in layer_name or 'output_blocks.3' in layer_name: # 4x downsample
            return h // 4, w // 4 # 10, 16
        # no such layers are exposed as cross-attn layers
        # if 'input_blocks.10' in layer_name or 'output_blocks.1' in layer_name: # 8x downsample
        #     return h // 8, w // 8
        # if 'input_blocks.11' in layer_name or 'output_blocks.0' in layer_name: # 8x downsample
        #     return h // 8, w // 8
        if 'middle_block' in layer_name:
            return h // 8, w // 8 # 5, 8
            
        # Default to a reasonable guess if no match
        return h // 4, w // 4


    def calculate_spatial_loss(self, attn_maps, bboxes, P):
        """
        Calculates the spatial loss based on attention maps and bounding boxes.
        """
        loss = torch.tensor(0.0, device=self.model.device)
        num_maps = 0
        
        # TODO: fix the P to be ~40% of the total number of pixels in the downsampled featuremap

        for layer_name, attn_map_raw in attn_maps.items():
            bsz, num_heads, total_frames, height, width, num_tokens = attn_map_raw.shape
            
            P = int(height * width * 1)
            
            # Average over heads and text tokens to get spatial attention
            attn_map_averaged = attn_map_raw.mean(dim=1) # (bsz, total_frames, height, width, num_tokens)
            attn_map_averaged = attn_map_averaged.squeeze(0) # (total_frames, height, width, num_tokens)
            attn_map = attn_map_averaged[:, :, :, 1] # (total_frames, height, width)
            
            # Get resolution for this layer
            h, w = height, width
            num_frames = total_frames

            print(f"[DEBUG] bboxes: {bboxes}\n\n")


            for f_idx in range(num_frames):
                frame_key = str(f_idx)
                if frame_key in bboxes:
                    curr_frame_attn_map = attn_map[f_idx, :, :]
                    
                    # TODO: enable parameterized token selection
                    # now: only select the 1st token
                    
                    bbox = bboxes[frame_key] # [x_min, y_min, x_max, y_max] in relative coords [0,1]
                    
                    # Downscale bbox to feature map size
                    x_min, y_min, x_max, y_max = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)

                    mask = torch.zeros_like(curr_frame_attn_map)
                    
                    
                    if y_max > y_min and x_max > x_min:
                        mask[y_min:y_max, x_min:x_max] = 1
                    
                    # Inside bbox
                    attn_inside = curr_frame_attn_map[mask == 1]
                    # Outside bbox
                    attn_outside = curr_frame_attn_map[mask == 0]
                    
                    print(f"[DEBUG] check attn_inside shape: {attn_inside.shape}")
                    print(f"[DEBUG] check attn_outside shape: {attn_outside.shape}")

                    # Loss for enhancing activations inside bbox
                    P_inside = int(attn_inside.numel() * 1)
                    top_p_inside, _ = torch.topk(attn_inside, P_inside)
                    loss_inbox = 1 - torch.mean(top_p_inside)
                    
                    # Loss for suppressing activations outside bbox
                    P_outside = int(attn_outside.numel() * 0.95)
                    top_p_outside, _ = torch.topk(attn_outside, P_outside)
                    loss_outbox = torch.mean(top_p_outside)

                    loss += (loss_inbox + loss_outbox)

                    num_maps += 1
        
        return loss / num_maps if num_maps > 0 else torch.tensor(0.0, device=self.model.device)


    def optimize(self, latent, cond, ts, unconditional_conditioning, unconditional_guidance_scale, uc_type=None, lr_scale_ratio=1.0):
        """
        Optimizes the latent representation.
        """
        # Make latent require grad
        latent_opt = latent.clone().detach().requires_grad_(True)

        # Setup optimizer
        print(f"lr_scale_ratio: {lr_scale_ratio}")
        optimizer = torch.optim.AdamW([latent_opt], lr=self.optim_params['lr'] * lr_scale_ratio)

        # Identify target transformer blocks and temporarily disable checkpointing
        target_blocks = []
        original_checkpoint_states = []
        for layer_name in self.optim_params['optim_ref_layers']:
            # layer_name is like 'input_blocks.8.1.attn2'
            # we need to get the parent 'input_blocks.8.1'
            block_name = '.'.join(layer_name.split('.')[:-1])
            try:
                block = self.model.model.diffusion_model.get_submodule(block_name)
                if hasattr(block, 'checkpoint'):
                    target_blocks.append(block)
                    original_checkpoint_states.append(block.checkpoint)
                    block.checkpoint = False
            except AttributeError:
                print(f"Warning: Could not find block {block_name} to disable checkpointing.")

        try:
            # Optimization loop
            print("Starting latent optimization...")
            with torch.enable_grad():
                for epoch in range(self.optim_params['epochs']):
                    optimizer.zero_grad()

                    # Forward pass through the model to get attention maps.
                    # We must not update the model's parameters.
                    self.model.zero_grad()

                    if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                        e_t = self.model.apply_model(latent_opt, ts, cond)
                    else:
                        if isinstance(cond, torch.Tensor):
                            e_t = self.model.apply_model(latent_opt, ts, cond)
                            e_t_uncond = self.model.apply_model(latent_opt, ts, unconditional_conditioning)
                        elif isinstance(cond, dict):
                            e_t = self.model.apply_model(latent_opt, ts, cond)
                            e_t_uncond = self.model.apply_model(latent_opt, ts, unconditional_conditioning)
                        else:
                            raise ValueError(f"Unsupported condition type: {type(cond)}")    
                        
                        
                    # # Replicating the classifier-free guidance logic from p_sample_ddim
                    # if unconditional_guidance_scale != 1.0 and unconditional_conditioning is not None:
                    #     # First, the unconditional pass
                    #     self.model.apply_model(latent_opt, ts, unconditional_conditioning)
                    #     # Then, the conditional pass. The attention maps from this pass are what we need.
                    #     self.model.apply_model(latent_opt, ts, cond)
                    # else:
                    #     self.model.apply_model(latent_opt, ts, cond)

                    # After the forward pass(es), attention maps are stored in the modules.
                    attn_maps = self.get_attention_maps(self.optim_params['optim_ref_layers'])
                    
                    # print(f"[DEBUG] extracted attn_maps: {attn_maps.keys()}")
                    
                    # Calculate loss
                    loss = self.calculate_spatial_loss(
                        attn_maps,
                        self.optim_params['bbox'],
                        self.optim_params['P']
                    )

                    if loss.requires_grad:
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        print(f"Epoch {epoch+1}/{self.optim_params['epochs']}, Loss: {loss.item()}")
                    else:
                        print(f"Epoch {epoch+1}/{self.optim_params['epochs']}, Loss: {loss.item()} (not a valid gradient)")
                        break
        finally:
            # Restore original checkpointing states
            for block, state in zip(target_blocks, original_checkpoint_states):
                block.checkpoint = state


        print("Latent optimization finished.")
        return latent_opt.detach() 