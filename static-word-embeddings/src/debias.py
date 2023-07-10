import sys
import json
import os

import numpy as np
from utils import load_embedding, load_pairs, filter_pairs
import config
sys.path.append("../external_code/bolukbasi/debiaswe/")
from debiaswe.we import WordEmbedding
from debiaswe.debias import debias

from sklearn.decomposition import PCA


class Debias:
    def __init__(self, vanilla_model_path, debiased_dir):
        """
        vanilla_model_path : path to the word embedding model to be debiased
        debiased_dir : path to dir to save the debiased models
        """
        self.vanilla_model_path = vanilla_model_path
        self.vanilla_model = load_embedding(vanilla_model_path)
        self.debiased_dir = debiased_dir
        self.model_name = vanilla_model_path.split("/")[-1]
        self.gendered_pairs = load_pairs(config.gender_pair_path)

    def prepare_hard_debias(self, E, verbose=0):
        """
        Loads related words and removed OOV for hard debiasing

        related words are
        1. gender definitional pairs: set of pairs of words that define gender dimension (e.g. he/she)
        2. equalize_pairs: set of gendered pairs that we'd like to equalize (e.g. spokesman/spokeswoman)
        3. gender specific seed: set of gendered words that we'd like to keep the gender of (e.g actress)
        """

        with open(config.bolukbasi_dir + 'data/definitional_pairs.json', "r") as f:
            defs = self.filter_pairs_HD(json.load(f), E)

        with open(config.bolukbasi_dir + 'data/equalize_pairs.json', "r") as f:
            equalize_pairs = self.filter_pairs_HD(json.load(f), E)

        with open(config.bolukbasi_dir + 'data/gender_specific_seed.json', "r") as f:
            gender_specific_words = self.filter_words_HD(json.load(f), E)

        # print("gender specific", len(gender_specific_words), gender_specific_words[:10])
        if verbose:
            print("definitional:\n", defs)
            print("equlize:\n", equalize_pairs)
            print("gender_specific:\n", gender_specific_words)

        return defs, equalize_pairs, gender_specific_words

    def filter_pairs_HD(self, pairs, E):
        """
        removes pairs (tuples) that are not in embedding E's vocab
        """
        count = 0
        for pair in pairs:
            if pair[0] not in E.words or pair[1] not in E.words:
                count+=1
                pairs.remove(pair)
        print("{} pairs removed".format(count))
        return pairs

    def filter_words_HD(self, words, E):
        """
        removes words that are not in embedding E's vocab
        """
        count = 0
        for word in words:
            if word not in E.words:
                words.remove(word)
                count+=1
        print("{} words removed".format(count))
        return words

    def hard_debias(self, save_dir=None):
        """
        This method runs the hard debiasing method according to Bolukbasi et al. Neurips 2016 paper.
        The code uses author's implementation of the method
        """
        E = WordEmbedding(self.vanilla_model_path)

        # loading the required data from hard debiasing --> using the released word sets
        defs, equalize_pairs, gender_specific_words = self.prepare_hard_debias(E)
        # running debiasing

        debias(E, gender_specific_words, defs, equalize_pairs)

        # saving and reloading the model into gensim format
        # print(self.vanilla_model_path)
        # print("model_name", self.model_name)
        if save_dir == None:
            save_dir = self.debiased_dir
        debiased_filename = os.path.join(config.debiased_model_dir , "HD_"+ str(self.model_name.split(".")[0]) + ".bin")
        E.save_w2v(debiased_filename)
        # print("saved in", debiased_filename)
        return load_embedding(debiased_filename)


    def do_pca(self, embedding, word_pairs):
        """
        dimensionality reduction to find representative space for the concept defined by word pairs (e.g. gender defined by gendered pairs such as he/she)
        runs pca on vectors defined by embeddings[word_pair[0]] - embedding[word_pair[1]]
        returns the fitted pca object
        """

        concept_vecs = []
        for index, pair in enumerate(word_pairs):
            # check if all exist in embedding:
            if pair[0] not in embedding.index_to_key or pair[1] not in embedding.index_to_key:
                print(pair, "not found in model", self.model_name)
                continue
            concept_vecs.append(np.array(embedding[pair[1]] - embedding[pair[0]]))
        concept_matrix = np.stack(concept_vecs, axis=0)
        # print(concept_matrix.shape)
        pca = PCA()
        pca.fit(concept_matrix)
        return pca

    def subtraction(self, embedding=None, def_pairs=None, save_dir=None):
        """
        debiasing method that subtracts the gender dimension from all vectors
        """
        if embedding == None:
            embedding = load_embedding(self.vanilla_model_path)

        if def_pairs == None:
            def_pairs = self.gendered_pairs
        else:
            def_pairs = filter_pairs(embedding=embedding, pairs=def_pairs)

        pca = self.do_pca(embedding, def_pairs)
        v_B = pca.components_[0]
        embedding.vectors = (embedding.vectors - v_B)

        sub_embed_filename = "sub_" + self.model_name.split(".")[0] + ".wv"
        if save_dir == None:
            save_dir = config.debiased_model_dir
        embedding.save(os.path.join(save_dir , sub_embed_filename))
        return embedding

    def projection(self, embedding=None, def_pairs=None, save_dir=None):
        """
        debiasing method that removes the projection of gender from every word embedding
        """

        if embedding == None:
            embedding = load_embedding(self.vanilla_model_path)

        if def_pairs == None:
            def_pairs = self.gendered_pairs

        def_pairs = filter_pairs(embedding=embedding, pairs=def_pairs)

        pca = self.do_pca(embedding, def_pairs)
        v_B = pca.components_[0]
        dot_prods = np.expand_dims(np.dot(embedding.vectors, v_B), axis=1)
        embedding.vectors = embedding.vectors - (dot_prods * v_B)
        prj_embed_filename = "prj_" + self.model_name.split(".")[0] + ".wv"
        self.prj = embedding
        if save_dir == None:
            save_dir = config.debiased_model_dir
        embedding.save(os.path.join(save_dir , prj_embed_filename))

        return embedding

    def partial_project(self, embedding=None, def_pairs=None):
        if embedding == None:
            embedding = load_embedding(self.vanilla_model_path)
        def f1(eta, sigma=1):
            return sigma / ((eta + 1) ** 2)

        def f2( eta, sigma=1):
            return np.exp(-(eta ** 2) / (sigma ** 2))

        def f3( eta, sigma=1):
            return np.maximum([0] * len(eta), sigma / (2 * eta))
        f = f1
        if def_pairs == None:
            def_pairs = self.gendered_pairs

        def_pairs = filter_pairs(embedding=embedding, pairs=def_pairs)

        mu = np.mean(np.array([(embedding[pair[0]] + embedding[pair[1]]) / 2 for pair in def_pairs]), axis=0)
        pca = self.do_pca(embedding, def_pairs)
        # TODO: check PCs : plot_component_vars(pca, def_pairs)
        v_B = pca.components_[0]
        beta = np.dot(embedding.vectors, v_B) - np.dot(mu, v_B)
        dot_prods = np.expand_dims(np.dot(embedding.vectors, v_B), axis=1)
        residual = embedding.vectors - (dot_prods * v_B)
        eta = np.linalg.norm(residual, axis=1)
        embedding.vectors = mu + residual + np.expand_dims(beta, axis=1) * (np.expand_dims(f(eta), axis=1) * v_B)
        return embedding


    def run_partial_projections(self, def_pairs=None, save_dir=None, run_all=True):

        if def_pairs == None:
            def_pairs = self.gendered_pairs

        func_dict = {"f1":self.f1, "f2":self.f2, "f3":self.f3}
        for func_name in func_dict:
            out_filename = "partial_prj_" + func_name + "_" + self.model_name.split(".")[0] + ".wv"

            embedding = load_embedding(self.vanilla_model_path)
            embedding = self.partial_project(embedding, func_dict[func_name], def_pairs)
            if save_dir == None:
                save_dir = config.debiased_model_dir
            embedding.save(os.path.join(save_dir, out_filename))
            del embedding
            print("saved {}".format(out_filename))
            if not run_all:
                break


    def get_vanilla_model(self):
        return self.vanilla_model


