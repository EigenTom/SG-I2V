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
            if module_name == 'CrossAttention' and 'attn2' in name and name in layer_names:
                # The attention map is saved in the module during the forward pass.
                # In `lvdm.modules.attention.CrossAttention`, `self.attn_map` is stored.
                # Its shape is (batch, heads, sequence_length_q, sequence_length_k).
                # For our case, this is (b*f, num_heads, h*w, num_tokens).
                if hasattr(module, 'attn_map') and module.attn_map is not None:
                    attn_maps[name] = module.attn_map
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
        h, w = self.model.image_size[0], self.model.image_size[1]
        
        if 'input_blocks.4' in layer_name or 'output_blocks.7' in layer_name: # 2x downsample
            return h // 2, w // 2
        if 'input_blocks.5' in layer_name or 'output_blocks.6' in layer_name: # 2x downsample
            return h // 2, w // 2
        if 'input_blocks.7' in layer_name or 'output_blocks.4' in layer_name: # 4x downsample
            return h // 4, w // 4
        if 'input_blocks.8' in layer_name or 'output_blocks.3' in layer_name: # 4x downsample
            return h // 4, w // 4
        if 'input_blocks.10' in layer_name or 'output_blocks.1' in layer_name: # 8x downsample
            return h // 8, w // 8
        if 'input_blocks.11' in layer_name or 'output_blocks.0' in layer_name: # 8x downsample
            return h // 8, w // 8
        if 'middle_block' in layer_name:
            return h // 8, w // 8
            
        # Default to a reasonable guess if no match
        return h // 4, w // 4


    def calculate_spatial_loss(self, attn_maps, bboxes, P):
        """
        Calculates the spatial loss based on attention maps and bounding boxes.
        """
        loss = torch.tensor(0.0, device=self.model.device)
        num_maps = 0

        for layer_name, attn_map_raw in attn_maps.items():
            # attn_map_raw shape: (b*f, num_heads, h*w, num_tokens)
            # Average over heads and text tokens to get spatial attention
            attn_map = torch.mean(attn_map_raw, dim=(1, 3)) # -> (b*f, h*w)

            # Get resolution for this layer
            h, w = self._get_layer_resolution(layer_name)
            
            if attn_map.shape[1] != h * w:
                # Fallback if resolution mismatch
                # This can happen if my _get_layer_resolution is not perfect.
                # Let's try to infer from shape
                inferred_hw = attn_map.shape[1]
                # Assuming square feature maps for simplicity if not easily factorizable
                h = w = int(inferred_hw ** 0.5)
                if h * w != inferred_hw:
                    # Not square, this is tricky. Let's skip if we can't determine.
                    print(f"Warning: could not determine shape for layer {layer_name}, skipping loss calculation for this map.")
                    continue

            # Reshape to (num_frames, h, w) - assuming batch size is 1 for inference
            num_frames = self.model.temporal_length
            try:
                attn_map = attn_map.reshape(num_frames, h, w)
            except RuntimeError:
                print(f"Warning: could not reshape attn_map for {layer_name}, skipping.")
                continue

            for f_idx in range(num_frames):
                frame_key = str(f_idx)
                if frame_key in bboxes:
                    bbox = bboxes[frame_key] # [x_min, y_min, x_max, y_max] in relative coords [0,1]
                    
                    # Downscale bbox to feature map size
                    x_min, y_min, x_max, y_max = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)

                    mask = torch.zeros_like(attn_map[f_idx])
                    if y_max > y_min and x_max > x_min:
                        mask[y_min:y_max, x_min:x_max] = 1
                    
                    # Inside bbox
                    attn_inside = attn_map[f_idx][mask == 1]
                    # Outside bbox
                    attn_outside = attn_map[f_idx][mask == 0]

                    # Loss for enhancing activations inside bbox
                    if attn_inside.numel() > P:
                        top_p_inside, _ = torch.topk(attn_inside, P)
                        loss = loss - torch.mean(top_p_inside)
                    
                    # Loss for suppressing activations outside bbox
                    if attn_outside.numel() > P:
                        top_p_outside, _ = torch.topk(attn_outside, P)
                        loss = loss + torch.mean(top_p_outside)

                    num_maps += 1
        
        return loss / num_maps if num_maps > 0 else torch.tensor(0.0, device=self.model.device)


    def optimize(self, latent, cond, ts, unconditional_conditioning, unconditional_guidance_scale):
        """
        Optimizes the latent representation.
        """
        # Make latent require grad
        latent_opt = latent.clone().detach().requires_grad_(True)

        # Setup optimizer
        optimizer = torch.optim.AdamW([latent_opt], lr=self.optim_params['lr'])

        # Optimization loop
        print("Starting latent optimization...")
        for epoch in range(self.optim_params['epochs']):
            optimizer.zero_grad()

            # Forward pass through the model to get attention maps.
            # We must not update the model's parameters.
            self.model.zero_grad()

            # Replicating the classifier-free guidance logic from p_sample_ddim
            if unconditional_guidance_scale != 1.0 and unconditional_conditioning is not None:
                # First, the unconditional pass
                self.model.apply_model(latent_opt, ts, unconditional_conditioning)
                # Then, the conditional pass. The attention maps from this pass are what we need.
                self.model.apply_model(latent_opt, ts, cond)
            else:
                self.model.apply_model(latent_opt, ts, cond)

            # After the forward pass(es), attention maps are stored in the modules.
            attn_maps = self.get_attention_maps(self.optim_params['optim_ref_layers'])
            
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


        print("Latent optimization finished.")
        return latent_opt.detach() 