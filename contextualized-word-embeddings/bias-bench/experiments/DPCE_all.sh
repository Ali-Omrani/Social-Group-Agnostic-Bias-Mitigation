model_name_or_path=$1
# model_name_or_path=bert-large-uncased


---------------------------stereoset---------------------------
python stereoset.py \
--model BertForMaskedLM \
--model_name_or_path $model_name_or_path

python stereoset_evaluation.py --predictions_dir ../results/stereoset


# ---------------------------CrowsPair---------------------------
python crows.py \
--bias_type religion \
--model_name_or_path $model_name_or_path

python crows.py \
--bias_type gender \
--model_name_or_path $model_name_or_path

python crows.py \
--bias_type race \
--model_name_or_path $model_name_or_path

# ---------------------------SEAT---------------------------
python seat.py --model_name_or_path $model_name_or_path

