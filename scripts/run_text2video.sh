name="base_512_v2"

ckpt='SG-I2V/checkpoints/base_512_v2/model.ckpt'
config='SG-I2V/configs/inference_t2v_512_v2.0.yaml'

prompt_file="SG-I2V/prompts/freetraj/text.txt"
res_dir="SG-I2V/results"

python3 SG-I2V/scripts/evaluation/inference_single.py \
--seed 42 \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 0.0 \
--prompt_file $prompt_file \
--fps 14
