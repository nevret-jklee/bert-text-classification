PLM="klue/roberta-base" # e.g. "klue/roberta-base"

# replace / with _ in BACKBONE
# FILE_NAME="kistep-${PLM//\//-}"
FILE_NAME="acc_final_preproc"
DIR_PATH="/data/nevret/bert-finetuning-custom/bert-text-classification"

python finetune.py \
    --file_name $FILE_NAME \
    --plm $PLM \
    --dir_path $DIR_PATH \
    --seed 42 \
    --learning_rate 2e-5 \
    --epochs 5 \
    --batch_size 32 \