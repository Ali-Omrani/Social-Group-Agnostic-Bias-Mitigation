debias_model=$1 
model_name_or_path=$2 

glue_tasks=(
    "rte"
    "wnli"
    "mrpc"
    "sst2"
    "stsb"
)

declare -A per_task_train_batch_size=(
    ["mrpc"]="16"
    ["mnli"]="32"
    ["cola"]="16"
    ["qnli"]="16"
    ["qqp"]="16"
    ["rte"]="16"
    ["sst2"]="16"
    ["stsb"]="16"
    ["wnli"]="10"
)
# learning_rate=(
#     "1e-5"
#     "2e-5"
#     "3e-5"
#     "4e-5"
#     "5e-5"
#     "1e-6"
#     "2e-6"
#     "3e-6"
#     "4e-6"
#     "5e-6"
#     "6e-6"
#     "7e-6"
#     "8e-6"
#     "9e-6"
# )

lr=5e-5

# for lr in ${learning_rate[@]}; do
    for task in ${glue_tasks[@]}; do
        if [ ! -f "glue_results/adept/${task}/eval_results.json" ]; then
            echo "===================================================="
            echo "===================================================="
            echo "===================================================="
            echo "Running ${task}"
            echo "===================================================="
            echo "===================================================="
            echo "===================================================="

            python run_glue.py \
                --model "BertPrefixForSequenceClassification" \
                --model_name_or_path $model_name_or_path \
                --task_name ${task} \
                --do_train \
                --do_eval \
                --max_seq_length 128 \
                --per_device_train_batch_size 16 \
                --learning_rate ${lr} \
                --num_train_epochs 3 \
                --eval_accumulation_steps 16 \
                --output_dir "../results/glue/${debias_model}/${task}/${lr}"
        fi
    done
# done