#!/bin/bash
MODEL_PATH=${1:-"DAMO-NLP-SG/VideoLLaMA3-7B-Image"}
BENCHMARKS=${2:-"AI2D","ChartQA","DocVQA","MathVista","MMMU","MMMU_interleaved","OCRBench","GQA","RealWorldQA","MMMU_Pro","MMMU-Pro_interleaved","BLINK","MME","InfoVQA","MathVision","MathVerse"}

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
DATA_ROOT=/mnt/damovl/EVAL_BENCH/IMAGE
declare -A DATA_ROOTS

# mcqa
DATA_ROOTS["AI2D"]="$DATA_ROOT/ai2diagram"
DATA_ROOTS["ChartQA"]="$DATA_ROOT/ChartQA"
DATA_ROOTS["DocVQA"]="$DATA_ROOT/DocVQA"
DATA_ROOTS["MathVista"]="$DATA_ROOT/MathVista"
DATA_ROOTS["MMMU"]="$DATA_ROOT/MMMU"
DATA_ROOTS["OCRBench"]="$DATA_ROOT/OCRBench"
DATA_ROOTS["GQA"]="$DATA_ROOT/GQA"
DATA_ROOTS["MMMU_Pro"]="$DATA_ROOT/MMMU_Pro"
DATA_ROOTS["RealWorldQA"]="$DATA_ROOT/RealworldQA"
DATA_ROOTS["BLINK"]="$DATA_ROOT/BLINK"
DATA_ROOTS["MME"]="$DATA_ROOT/MME"
DATA_ROOTS["InfoVQA"]="$DATA_ROOT/InfoVQA"
DATA_ROOTS["MathVerse"]="$DATA_ROOT/MathVerse"
DATA_ROOTS["MathVision"]="$DATA_ROOT/MathVision"


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
        --max_visual_tokens 16384
done