import os

data_dir = "../data"
embedding_dir = os.path.join(data_dir, "embedding")
glove_path = os.path.join(embedding_dir, "glove.6B.300d.txt")
sgns_wikitext_path = os.path.join(embedding_dir, "31.model")
debiased_model_dir = "../models"
professions_path = os.path.join(data_dir, "resource_files", "professions.json")
gender_pair_path = os.path.join(data_dir, "resource_files", "gender_pairs.json")
race_pair_path = os.path.join(data_dir, "resource_files", "race_pairs.json")
age_pair_path = os.path.join(data_dir, "resource_files", "age_pairs.json")
warmth_pair_path = os.path.join(data_dir, "resource_files", "warmth_pairs.json")
competence_pair_path = os.path.join(data_dir, "resource_files", "competence_pairs.json")

weat_config_path = "../data/resource_files/attenuating_eval_weat.json"

bolukbasi_dir = "../external_code/bolukbasi/debiaswe/"