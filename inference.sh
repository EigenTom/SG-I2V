#Run sh ./inference.sh
for f in ./examples/*; do
    if [ -d "$f" ]; then
        # Will not run if no directories are available
        folder=$(basename -- "$f")
        CUDA_VISIBLE_DEVICES=8 python inference.py --input_dir "$f" --output_dir ./output/"$folder"
    fi
done