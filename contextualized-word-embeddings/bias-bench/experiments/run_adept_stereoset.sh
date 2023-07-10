#!/bin/bash
#SBATCH --nodelist=ink-titan
#SBATCH --time=1-0:00
#SBATCH --job-name=deprom
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1


cd /home/ali/contextualized-word-embeddings/bias-bench/experiments
conda activate biasbench

model_name_or_path=$1


if [ ! -f "../results/stereoset/ADEPTBertForMaskedLM.json" ]; then
    echo "Running stereoset"
    python stereoset_debias.py \
        --model "ADEPTBertForMaskedLM" \
        --model_name_or_path $model_name_or_path
fi

python stereoset_evaluation.py \
    --persistent_dir ".." \
    --predictions_dir "../results/stereoset" \
    --predictions_file "ADEPTBertForMaskedLM.json"

if [ ! -f "../results/stereoset/BertForMaskedLM.json" ]; then
    echo "Running stereoset"
    python stereoset.py \
        --model "BertForMaskedLM" \
        --model_name_or_path "bert-large-uncased"
fi

python stereoset_evaluation.py \
    --persistent_dir ".." \
    --predictions_dir "../results/stereoset" \
    --predictions_file "BertForMaskedLM.json"