model_name_or_path=$1

if [ ! -f "../results/crows/ADEPTBertForMaskedLM.json" ]; then
    echo "Running crows"
    python crows_debias.py \
        --model "ADEPTBertForMaskedLM" \
        --model_name_or_path $model_name_or_path
fi