#!/bin/sh

# created by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>
# updated on Dec. 22, 2020 by Katsuki Chousa <katsuki.chousa.bg at hco.ntt.co.jp>

PROJECT_DIR=__SET_DIR_PATH__
EXPERIMENT_DIR=$PROJECT_DIR/experiments
MODEL_TYPE=xlm-roberta
DATA_DIR=$PROJECT_DIR/data

if [ $# -ne 4 ]; then
    echo "$0 model_path output_dir TEST.json test_title"
    exit 1
fi

model_path=$1
test_file=$3
test_prefix=`basename $test_file .json`
output_dir=$2/$test_prefix
test_title=$4

date
hostname
echo $EXPERIMENT_DIR

echo ""
echo "### extraction ###"
mkdir -p $output_dir

if [ ! -e $output_dir/nbest_predictions_.json ]; then
    python $PROJECT_DIR/run_qa_alignment.py \
        --model_type $MODEL_TYPE \
        --model_name_or_path $model_path \
        --version_2_with_negative \
        --do_eval \
        --predict_file $test_file \
        --max_seq_length 384 \
        --max_query_length 158 \
        --max_answer_length 158 \
        --doc_stride 64 \
        --n_best_size 10 \
        --data_dir $output_dir \
        --output_dir $output_dir \
        --overwrite_output_dir \
        --save_steps 5000 \
        --per_gpu_eval_batch_size 240 \
        --thread 8 2>&1 \
    | tee $output_dir/span_prediction.log
fi

python $PROJECT_DIR/scripts/get_sent_align_for_overlap.py \
    --nbest 1 \
    $DATA_DIR/sample.{l1,l2} \
    $output_dir/nbest_predictions_.json \
    $test_title \
    $output_dir/$test_prefix \
