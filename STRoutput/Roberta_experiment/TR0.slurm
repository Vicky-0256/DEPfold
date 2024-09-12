#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4   # Modified to allocate 2 GPUs per task
#SBATCH --tasks-per-node=1  # Ensure there is only 1 task per node
#SBATCH --mem=320G 
#SBATCH --time=50:00:00 
#SBATCH --account=jlxi8926-auto-sum
#SBATCH --qos=epsrc
#SBATCH --output=STRoutput/Roberta_experiment/running.out

set -e
module purge; module load baskerville
module load Python/3.8.6-GCCcore-10.2.0
source ../mlp/bin/activate

# cd RNA_parser/

export CUDA_LAUNCH_BLOCKING=1


# 定义变量
OUTPUT_DIR="STRoutput/Roberta_experiment"
MODEL_PATH="$OUTPUT_DIR/model.pt"
LOSS_TYPE="focal_loss"  # focal_loss
EMBEDDING="roberta-base" # roberta-base
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE_TRAIN=8
EVAL_BATCH_SIZE_PREDICT=8
NUM_EPOCHS=100
EARLY_STOP_TRAIN=8
EARLY_STOP_PREDICT=100
CACHE_DATA="./data/Str_"
FINETUNE=True
ISPSE=True
TREE=False
PROJ=False
BETA=0.0

# 训练模型
python run_parser.py \
  --embedding "$EMBEDDING" \
  --finetune \
  --train_session "TR" \
  --eval_session "VL" \
  --test_session "TE" \
  --cache_data "$CACHE_DATA" \
  --early_stop "$EARLY_STOP_TRAIN" \
  --per_gpu_train_batch_size "$TRAIN_BATCH_SIZE" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_gpu_eval_batch_size "$EVAL_BATCH_SIZE_TRAIN" \
  --mode "train" \
  --output_dir "$OUTPUT_DIR" \
  --loss "$LOSS_TYPE" \
  --beta "$BETA" \
  --tree \
  --proj \

# 预测 Session TS0
PREDICT_SESSION_TS0="TE"
PREDICT_SAVE_TS0="$OUTPUT_DIR/TE/"
python run_parser.py \
  --beta "$BETA" \
  --mode predict \
  --output_dir "$OUTPUT_DIR" \
  --predict_session "$PREDICT_SESSION_TS0" \
  --predict_save "$PREDICT_SAVE_TS0" \
  --path "$MODEL_PATH" \
  --loss "$LOSS_TYPE" \
  --embedding "$EMBEDDING" \
  --early_stop "$EARLY_STOP_PREDICT" \
  --per_gpu_train_batch_size "$TRAIN_BATCH_SIZE" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_gpu_eval_batch_size "$EVAL_BATCH_SIZE_PREDICT" \
  --cache_data "$CACHE_DATA" \
  --tree \
  --proj \

# 预测 Session bpnew
PREDICT_SESSION_NEWBP="AR"
PREDICT_SAVE_NEWBP="$OUTPUT_DIR/AR/"
python run_parser.py \
  --beta "$BETA" \
  --mode predict \
  --output_dir "$OUTPUT_DIR" \
  --predict_session "$PREDICT_SESSION_NEWBP" \
  --predict_save "$PREDICT_SAVE_NEWBP" \
  --path "$MODEL_PATH" \
  --loss "$LOSS_TYPE" \
  --embedding "$EMBEDDING" \
  --early_stop "$EARLY_STOP_PREDICT" \
  --per_gpu_train_batch_size "$TRAIN_BATCH_SIZE" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_gpu_eval_batch_size "$EVAL_BATCH_SIZE_PREDICT" \
  --cache_data "$CACHE_DATA" \
  --tree \
  --proj \


# # 评估 TS0 预测结果

# python evaluate.py "$PREDICT_SAVE_TS0"

# # 评估 newbp 预测结果
# python evaluate.py "$PREDICT_SAVE_NEWBP"
