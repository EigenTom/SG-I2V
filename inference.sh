#Run sh ./inference.sh
for f in ./examples/*; do
    if [ -d "$f" ]; then
        folder=$(basename -- "$f")
        CUDA_VISIBLE_DEVICES=0 python inference.py --input_dir "$f" --output_dir ./output/"$folder"
    fi
done