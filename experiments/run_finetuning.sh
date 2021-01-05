#!/bin/sh

# created by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>
# updated on Dec. 22, 2020 by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>

PROJECT_DIR=__SET_DIR_PATH__
EXPERIMENT_DIR=$PROJECT_DIR/experiments
OUTPUT_DIR=$EXPERIMENT_DIR/finetuning

DATA_DIR=$PROJECT_DIR/data
TRAIN_FILE=$DATA_DIR/train.json
DEV_FILE=$DATA_DIR/dev.json

MODEL_TYPE=xlm-roberta
MODEL_NAME=xlm-roberta-base

date
hostname
echo $EXPERIMENT_DIR

echo ""
echo "### finetuning ###"
mkdir -p $OUTPUT_DIR
python $PROJECT_DIR/run_qa_alignment.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --version_2_with_negative \
    --do_train \
    --do_eval \
    --eval_all_checkpoints \
    --train_file $TRAIN_FILE \
    --predict_file $DEV_FILE \
    --learning_rate 3e-5 \
    --per_gpu_train_batch_size 5 \
    --num_train_epochs 5 \
    --max_seq_length 384 \
    --max_query_length 158 \
    --max_answer_length 158 \
    --doc_stride 64 \
    --n_best_size 10 \
    --data_dir $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --save_steps 5000 \
    --thread 4 2>&1 \
| tee $EXPERIMENT_DIR/finetuning.log
