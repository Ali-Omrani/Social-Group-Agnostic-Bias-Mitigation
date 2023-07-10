model_name_or_path=$1


seat_tests="sent-religion1 "\
"sent-religion1b "\
"sent-religion2 "\
"sent-religion2b "\
"sent-angry_black_woman_stereotype "\
"sent-angry_black_woman_stereotype_b "\
"sent-weat3 "\
"sent-weat3b "\
"sent-weat4 "\
"sent-weat5 "\
"sent-weat5b "\
"sent-weat6 "\
"sent-weat6b "\
"sent-weat7 "\
"sent-weat7b "\
"sent-weat8 "\
"sent-weat8b"

if [ ! -f "../results/seat/ADEPTBertModel.json" ]; then
    echo "Running seat"
    python seat_debias.py \
        --tests ${seat_tests} \
        --model "ADEPTBertModel" \
        --model_name_or_path $model_name_or_path
fi