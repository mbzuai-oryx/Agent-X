#!/bin/bash
MODEL_PATH=${1:-"DAMO-NLP-SG/VideoLLaMA3-7B"}
BENCHMARKS=${2:-"mvbench,videomme,egoschema,perception_test,activitynet_qa,mlvu,longvideobench,tempcompass,nextqa,tomato,charades_sta,activitynet_tg"}

ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-8}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${6:-0}

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "BENCHMARKS: $BENCHMARKS"


SAVE_DIR=evaluation_results
DATA_ROOT=/mnt/damovl/EVAL_BENCH/VIDEO
declare -A DATA_ROOTS

# mcqa
DATA_ROOTS["mvbench"]="$DATA_ROOT/mvbench"
DATA_ROOTS["videomme"]="$DATA_ROOT/videomme"
DATA_ROOTS["egoschema"]="$DATA_ROOT/egoschema"
DATA_ROOTS["perception_test"]="$DATA_ROOT/perception_test"
DATA_ROOTS["activitynet_qa"]="$DATA_ROOT/activitynet_qa"
DATA_ROOTS["mlvu"]="$DATA_ROOT/mlvu"
DATA_ROOTS["longvideobench"]="$DATA_ROOT/longvideobench"
DATA_ROOTS["lvbench"]="$DATA_ROOT/lvbench"
DATA_ROOTS["tempcompass"]="$DATA_ROOT/tempcompass"
DATA_ROOTS["nextqa"]="$DATA_ROOT/nextqa"
DATA_ROOTS["charades_sta"]="$DATA_ROOT/charades"


IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS"
for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$BENCHMARK]}
    if [ -z "$DATA_ROOT" ]; then
        echo "Error: Data root for benchmark '$BENCHMARK' not defined."
        continue
    fi
    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        evaluation/evaluate.py \
        --model_path ${MODEL_PATH} \
        --benchmark ${BENCHMARK} \
        --data_root ${DATA_ROOT} \
        --save_path "${SAVE_DIR}/${MODEL_PATH##*/}/${BENCHMARK}.json" \
        --fps 1 \
        --max_frames 180 \
        --max_visual_tokens 16384
done