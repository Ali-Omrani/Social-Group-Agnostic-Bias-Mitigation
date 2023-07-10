model_type=$1
gpu=$2
algorithm=$3 # ADEPT ADEPT-finetuning DPCE
bias=$4 # gender religion scm
ab_test_type=final # raw reliability quality quantity-100 quantity-1000 quantity-10000
debias_layer=all # first last all
loss_target=token # token sentence
dev_data_size=1000
seed=${5:-42}
train_data_size=$5

if [ $model_type == 'bert' ]; then
    model_name_or_path=bert-large-uncased
elif [ $model_type == 'roberta' ]; then
    model_name_or_path=roberta-large
fi

TRAIN_DATA=../sentences_collection/$model_name_or_path/$bias/$ab_test_type/data.bin
OUTPUT_DIR=../debiased_models/$seed/$model_name_or_path/$algorithm/$bias/$ab_test_type/$train_data_size
LOG_DIR=../log/$algorithm/$bias/$ab_test_type

rm -r $OUTPUT_DIR
echo $model_type $algorithm $bias $ab_test_type $seed

if [ $bias == 'scm' ]; then
    program=../scm_debias.py
else
    program=../debias.py
fi

if [ $algorithm == 'ADEPT' -o $algorithm == 'ADEPT-finetuning' -o $algorithm=='ADEPT-DPCEloss' ]; then
    alpha=0.3
    beta=0.7
    perplexity=15
    CUDA_VISIBLE_DEVICES=$gpu python $program \
        --algorithm=$algorithm \
        --bias=$bias \
        --data_file=$TRAIN_DATA \
        --output_dir=$OUTPUT_DIR \
        --log_dir=$LOG_DIR \
        --model_type=$model_type \
        --model_name_or_path=$model_name_or_path \
        --seed $seed \
        --do_train \
        --dev_data_size $dev_data_size \
        --do_eval \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 8 \
        --num_train_epochs 20 \
        --block_size 128 \
        --loss_target $loss_target \
        --debias_layer $debias_layer \
        --weighted_loss $alpha $beta \
        --perplexity $perplexity \
        --use_neutral true

elif [ $algorithm == 'DPCE' ]; then
    alpha=0.2
    beta=0.8
    CUDA_VISIBLE_DEVICES=$gpu python ../debias.py \
        --algorithm=$algorithm \
        --bias=$bias \
        --data_file=$TRAIN_DATA \
        --output_dir=$OUTPUT_DIR \
        --log_dir=$LOG_DIR \
        --model_type=$model_type \
        --model_name_or_path=$model_name_or_path \
        --seed $seed \
        --do_train \
        --dev_data_size $dev_data_size \
        --train_data_size $train_data_size \
        --do_eval \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 8 \
        --num_train_epochs 10 \
        --block_size 128 \
        --loss_target $loss_target \
        --debias_layer $debias_layer \
        --weighted_loss $alpha $beta \

fi


