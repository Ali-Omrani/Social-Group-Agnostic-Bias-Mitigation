import gensim
import json
import sys
import inflect
import nltk
nltk.download('wordnet')
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import rankdata, spearmanr
from utils import load_pairs

class Evalution:
    def __init__(self, profession_file_path, weat_config_path=None, model_path=None):
        self.profession_path = profession_file_path
        self.weat_config_path = weat_config_path
        self.gendered_pairs = [('man', 'woman'), ('son', 'daughter'), ('he', 'she'), ('male', 'female'),
                               ('boy', 'girl'), ('himself', 'herself'), ('guy', 'gal'), ('father', 'mother'),
                               ('john', 'mary')]
        self.model_path = model_path
        if model_path:
            self.embedding = weat.load_embedding(model_path)
        self.extended_professions = {}

    def test_weat(self, embedding=None):
        if not embedding:
            embedding = self.embedding
        weat_df = pd.DataFrame(columns=["test", "effect_size", "err"])
        with open(self.weat_config_path) as cf:
            config = json.load(cf)
            for name_of_test, test_config in config['tests'].items():
                for key in test_config:
                    test_config[key] = [item.lower() for item in test_config[key]]
                mean, err = weat.run_test(test_config, embedding)
                weat_df = weat_df.append({"test": name_of_test, "effect_size": mean, "err": err}, ignore_index=True)
        return weat_df


    def filter_words(self, embedding, words):
        return [w for w in words if w in embedding.index_to_key]

    def filter_pairs(self, embedding, pairs):
        for pair in pairs:
            if pair[0] not in embedding.index_to_key or pair[1] not in embedding.index_to_key:
                pairs.remove(pair)
                print("Pair {} removed -- OOV".format(pair))
        return pairs

    def load_professions(self):
        with open(self.profession_path) as professions_file:
            professions = json.load(professions_file)
        professions = [p[0].strip().lower() for p in professions]
        return professions

    def ect(self, embedding=None, pairs=None, ect_type="terms"):
        if not embedding:
            embedding = self.embedding

        # loading profession words
        professions = self.filter_words(embedding, self.load_professions())

        # defining gender pairs
        if pairs:
            gendered_pairs = self.filter_pairs(embedding, pairs)
        else:
            gender_dict = {}
            gender_dict["terms"] = self.gendered_pairs
            gendered_name_pairs = []

            gendered_pairs = gender_dict[ect_type]
            gendered_pairs = self.filter_pairs(embedding, gendered_pairs)
        male_mean = np.mean([embedding[pair[0]] for pair in gendered_pairs], axis=0)
        female_mean = np.mean([embedding[pair[1]] for pair in gendered_pairs], axis=0)
        u_m = [cosine(embedding[p], male_mean) for p in professions]
        u_f = [cosine(embedding[p], female_mean) for p in professions]
        return spearmanr(rankdata(u_m), rankdata(u_f))

    def get_syn_and_plurals(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        len(set(synonyms))

        engine = inflect.engine()
        plurals = [engine.plural(syn) for syn in synonyms]
        return synonyms + plurals

    def eqt(self, embedding=None, pairs=None):
        if not embedding:
            embedding = self.embedding

        if pairs == None:
            gendered_pairs = self.gendered_pairs
        else:
            gendered_pairs = pairs

        EQT_dict = {}
        professions = self.filter_words(embedding, self.load_professions())
        if len(self.extended_professions)== 0:
            for profession in professions:
                self.extended_professions[profession]= self.get_syn_and_plurals(profession)
                self.extended_professions[profession].append(profession)
        gendered_pairs = self.filter_pairs(embedding, gendered_pairs)
        for pair in tqdm(gendered_pairs):
            EQT_dict[pair[0] + "-" + pair[1]] = 0
            for profession in professions:
                analogy = embedding.most_similar(positive=[pair[1], profession], negative=[pair[0]])
                if analogy[0][0] in self.extended_professions[profession]:
                    EQT_dict[pair[0] + "-" + pair[1]] += 1
            EQT_dict[pair[0] + "-" + pair[1]] /= len(professions)
        return EQT_dict
