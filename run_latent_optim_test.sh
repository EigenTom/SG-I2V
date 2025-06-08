python3 SG-I2V/scripts/evaluation/inference_single.py \
--config SG-I2V/configs/inference_t2v_512_v2.0.yaml \
--ckpt_path SG-I2V/checkpoints/base_512_v2/model.ckpt \
--prompt_file "./test_prompt.txt" \
--savedir "SG-I2V/results/optimized_run" \
--use_latent_optimization \
--optim_bbox_config "./bbox.json" \
--optim_k 10 \
--optim_lr 0.2 \
--optim_epochs 1 \
--optim_ref_layers \
    'input_blocks.8.1.transformer_blocks.0.attn2' \
    'input_blocks.5.1.transformer_blocks.0.attn2' \
    'input_blocks.7.1.transformer_blocks.0.attn2' \
    'input_blocks.2.1.transformer_blocks.0.attn2' \
    'input_blocks.4.1.transformer_blocks.0.attn2' \
    'input_blocks.1.1.transformer_blocks.0.attn2' \
    'output_blocks.3.1.transformer_blocks.0.attn2' \
    'output_blocks.4.1.transformer_blocks.0.attn2' \
    'output_blocks.5.1.transformer_blocks.0.attn2' \
    'output_blocks.6.1.transformer_blocks.0.attn2' \
    'output_blocks.7.1.transformer_blocks.0.attn2' \
    'middle_block.1.transformer_blocks.0.attn2' \
    'output_blocks.8.1.transformer_blocks.0.attn2' \
    'output_blocks.10.1.transformer_blocks.0.attn2' \
    'output_blocks.11.1.transformer_blocks.0.attn2' \
    