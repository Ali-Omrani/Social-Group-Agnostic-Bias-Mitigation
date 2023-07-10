debias_model=$1 #debiased_models_DPCE_gender_500
model_name_or_path=$2 

# glue_tasks=(
#     "mrpc"
#     "wnli"
#     "mnli"
#     "cola"
#     "qnli"
#     "qqp"
#     "rte"
#     "stsb"
#     "sst2"
# )
glue_tasks=(
    "rte"
    "wnli"
    "mrpc"
    "sst2"
    "stsb"
)

# learning_rate=(
#     "6e-6"
#     "7e-6"
#     "8e-6"
#     "9e-6"
# )
lr=5e-5

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

# for lr in ${learning_rate[@]}; do
    for task in ${glue_tasks[@]}; do
            echo "===================================================="
            echo "===================================================="
            echo "===================================================="
            echo "Running ${task}"
            echo "===================================================="
            echo "===================================================="
            echo "===================================================="
            python run_glue.py \
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
    done
# done
