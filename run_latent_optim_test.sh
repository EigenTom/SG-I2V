python3 ./scripts/evaluation/inference_single.py \
--config ./configs/inference_t2v_512_v2.0.yaml \
--ckpt_path ./checkpoints/model.ckpt \
--prompt_file "./test_prompt.txt" \
--savedir "./results/optimized_run" \
--use_latent_optimization \
--optim_bbox_config "./bbox_center.json" \
--optim_k 10 \
--optim_lr 0.02 \
--optim_epochs 4 \
--optim_ref_layers \
    'input_blocks.4.1.transformer_blocks.0.attn2' \
    'input_blocks.5.1.transformer_blocks.0.attn2' \
    'input_blocks.7.1.transformer_blocks.0.attn2' \
    'input_blocks.8.1.transformer_blocks.0.attn2' \
    'middle_block.1.transformer_blocks.0.attn2' \
    'output_blocks.3.1.transformer_blocks.0.attn2' \
    'output_blocks.4.1.transformer_blocks.0.attn2' \
    'output_blocks.5.1.transformer_blocks.0.attn2' \
    'output_blocks.6.1.transformer_blocks.0.attn2' \
    'output_blocks.7.1.transformer_blocks.0.attn2' \
    # 'input_blocks.1.1.transformer_blocks.0.attn2' \
    # 'input_blocks.2.1.transformer_blocks.0.attn2' \
    # 'output_blocks.8.1.transformer_blocks.0.attn2' \
    # 'output_blocks.10.1.transformer_blocks.0.attn2' \
    # 'output_blocks.11.1.transformer_blocks.0.attn2' \
# --optim_k 15 \
# --optim_lr 0.08 \
# lr too low: cannot ctrl object location, or produce mixed results
# lr too high: cause the generation collapse
# optim_k: high k need to be coupled with high lr, otherwise generation collapse
# higher lr might cause visual artifacts
# higher optim_k seems to be better controlling object shape/location
# if optim_k=15: with lr=0.09, can see an entire eagle
# if optim_k=10 with lr=0.6, we have eagle head
# tradeoff between high lr, single optim and low lr, multiple optim: 
# low lr, multiple optim seems to be have less visual artifacts