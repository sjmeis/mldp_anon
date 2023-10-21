# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.special import lambertw, softmax
import scipy.stats as sct
from scipy.linalg import sqrtm
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import numba as nb
import math

@nb.njit(fastmath=True,parallel=True)
def calc_distance(vec_1,vec_2):
    res = np.empty((vec_1.shape[0],vec_2.shape[0]),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            res[i,j] = np.sqrt((vec_1[i,0]-vec_2[j,0])**2+(vec_1[i,1]-vec_2[j,1])**2+(vec_1[i,2]-vec_2[j,2])**2)
    return res

def calc_probability(embed1, embed2, epsilon=2):
    distance = calc_distance(embed1, embed2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix

def euclidean_dt(m, v):
    diff = m - v
    dist = np.sum(np.square(diff), axis=-1)
    return np.sqrt(dist)

def truncated_Poisson(mu, max_value, size):
    temp_size = size
    while True:
        temp_size *= 2
        temp = sct.poisson.rvs(mu, size=temp_size)
        truncated = temp[temp <= max_value]
        if len(truncated) >= size:
            return truncated[:size]

def truncated_Gumbel(mu, scale, max_value, size):
    temp_size = size
    while True:
        temp_size *= 2
        temp = np.random.gumbel(loc = mu, size=temp_size, scale = scale.real)
        truncated = temp[np.absolute(temp) <= max_value]
        if len(truncated) >= size:
            return truncated[:size]
        
def euclidean_laplace_rand_fn(dimensions, epsilon):
    v = np.random.multivariate_normal(mean = np.zeros(dimensions),
                                        cov = np.eye(dimensions))
    v_norm = np.linalg.norm(v) + 1e-30
    v = v / v_norm

    l = np.random.gamma(shape = dimensions, scale = 1 / epsilon)
    return l * v

# adapted from https://github.com/xiangyue9607/SanText/blob/main/SanText.py
class SanText:
    def __init__(self, vocab_list, epsilon, embed_type, wv_model, embedding_matrix, dim):
        self.vocab_list = vocab_list
        self.epsilon = epsilon
        self.wv_model = wv_model
        self.embedding_matrix = embedding_matrix
        self.embed_type = embed_type
        self.dim = dim

        self.prob_matrix = calc_probability(self.embedding_matrix, self.embedding_matrix, epsilon=self.epsilon)
        self.prob_matrix = {x[0]:x[1] for x in zip(self.vocab_list, self.prob_matrix)}

    def replace_word(self, word):
        if word not in self.prob_matrix:
            return word

        sampling_prob = self.prob_matrix[word]
        sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
        if len(sampling_index) > 0:
            try:
                return self.vocab_list[sampling_index[0]]
            except IndexError:
                return word
        else:
            return word

    def SanText(self, doc):
        new_doc = []
        for token in doc:
            new_doc.append(self.replace_word(token))
        return new_doc
    
    def export_pm(self):
        return {x:self.prob_matrix[x].tolist() for x in self.prob_matrix}
##############################################################################

class MultivariateCalibrate:
    def __init__(self, vocab_dict, epsilon, embed_type, wv_model, embedding_matrix, dim):
        self.vocab_dict = vocab_dict
        self.epsilon = epsilon
        self.wv_model = wv_model
        self.embedding_matrix = embedding_matrix
        self.embed_type = embed_type
        self.dim = dim
        self.num_perturbed = 0
        self.num_words = 0

    def get_perturbed_vector(self, word_vec, n):
        noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
        norm_noise = noise / np.linalg.norm(noise)
        N = np.random.gamma(n, 1/self.epsilon) * norm_noise
        return word_vec + N

    def get_nearest(self, vector):
        diff = (self.embedding_matrix - vector)
        most_sim_index = np.argmin(np.linalg.norm(diff, axis=1))

        return most_sim_index

    def get_word_from_index(self, index):
        keys = [k for k, v in self.vocab_dict.items() if v == index]
        if keys:
            return keys[0]
        return None

    def replace_word(self, word):
        N = None
        if self.embed_type == "glove":
            embedding_vector = self.wv_model.get(word)

        elif self.embed_type == "word2vec":
            if word in self.wv_model: embedding_vector = self.wv_model[word]
            else: embedding_vector = None

        if embedding_vector is not None:
            self.num_words += 1
            perturbed_vector = self.get_perturbed_vector(embedding_vector, self.dim)
            sim_ind = self.get_nearest(perturbed_vector)
            new_word = self.get_word_from_index(sim_ind)
            if new_word is None:  return word
            else: 
                self.num_perturbed += 1
                return new_word
        return word

class TruncatedGumbel:
    def __init__(self, tokenizer, wv_model, 
                 embed_type, embedding_matrix,
                 epsilon, dim,
                 max_inter_dist=0,
                 min_inter_dist=np.inf):
        self.tokenizer = tokenizer
        self.embed_type = embed_type
        self.epsilon = epsilon
        self.wv_model = wv_model
        self.embedding_matrix = embedding_matrix
        self.dim = dim
        self.num_perturbed = 0
        self.num_words = 0

        self.max_inter_dist = max_inter_dist
        self.min_inter_dist = min_inter_dist

        if self.max_inter_dist == 0 or self.min_inter_dist == np.inf:
            for i in range(1, len(embedding_matrix)):
                dist = euclidean_dt(embedding_matrix[1:], embedding_matrix[i])
                dist = [x for x in dist if x > 0.0]
                idx = np.argsort(dist)
                mindist = dist[idx[0]]
                maxdist = dist[idx[len(dist) - 1]]

                if mindist < self.min_inter_dist:
                    self.min_inter_dist = mindist
                if maxdist > self.max_inter_dist:
                    self.max_inter_dist = maxdist

        #self.a = (self.epsilon - (2/self.min_inter_dist)*(1 + np.log(len(embedding_matrix[1:])))) / 3
        self.a = (self.epsilon - (2*(1 + np.log(len(embedding_matrix[1:]))) / self.min_inter_dist)) / 3

        if self.a * self.min_inter_dist <= 0:
            self.b = 2 * self.max_inter_dist / (lambertw(2 * self.a * self.max_inter_dist).real)
        else:
            self.b = 2 * self.max_inter_dist / (np.min(np.array([ lambertw(2 * self.a * self.max_inter_dist).real, np.log(self.a * self.min_inter_dist) ])) )
        
    def replace_word(self, word):
        self.num_words += 1      
        if self.embed_type == "glove":
            word_embed = self.wv_model.get(word)

        elif self.embed_type == "word2vec":
            if word in self.wv_model: word_embed = self.wv_model[word]
            else: word_embed = None

        if word_embed is None: 
            return word

        k = truncated_Poisson(mu = np.log(len(self.embedding_matrix) - 1), 
                              size = 1,
                              max_value = len(self.embedding_matrix) - 1)
        k = k[0]
        dist = euclidean_dt(self.embedding_matrix[1:], word_embed)
        idx = np.argsort(dist)
        idx = idx[:k]
        dist = dist[idx]
        dist = dist + truncated_Gumbel(mu = 0, scale = self.b, 
                                       size=len(dist), 
                                       max_value=self.max_inter_dist)
        indexes = np.argsort(dist)
        if len(indexes) > 0:
            i = idx[indexes[0]]
        else:
            return word      

        perturbed_word = self.tokenizer.index_word[i + 1]
        return perturbed_word    

class VickreyMechanism:
    def __init__(self, tokenizer, wv_model, 
                 embed_type, embedding_matrix,
                 epsilon, dim,
                 k = 2, t = [1, 0]):
        self.tokenizer = tokenizer
        self.embed_type = embed_type
        self.epsilon = epsilon
        self.wv_model = wv_model
        self.embedding_matrix = embedding_matrix
        self.dim = dim
        self.num_perturbed = 0
        self.num_words = 0
        self.k = k
        self.t = np.asarray(t)
    
    def replace_word(self, word):
        self.num_words += 1      
        if self.embed_type == "glove":
            word_embed = self.wv_model.get(word)

        elif self.embed_type == "word2vec":
            if word in self.wv_model: word_embed = self.wv_model[word]
            else: word_embed = None

        if word_embed is None: 
            return word

        noisy_vector = word_embed + euclidean_laplace_rand_fn(dimensions = len(word_embed), epsilon = self.epsilon)
        dist = euclidean_dt(self.embedding_matrix[1:], noisy_vector)

        idx = np.argsort(dist)
        idx = idx[1:] # w \belongsto W - {input word}
        idx = idx[:self.k]
        dist = dist[idx]

        p = -self.t * dist
        p = softmax(p)

        i = np.random.choice(idx, p = p)

        perturbed_word = self.tokenizer.index_word[i + 1]
        return perturbed_word

class TEM:
    def __init__(self, vocab_dict, epsilon, embed_type, wv_model, embedding_matrix, vocab_size, dim):
        self.vocab_dict = vocab_dict
        self.embed_type = embed_type
        self.epsilon = epsilon
        self.wv_model = wv_model
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_perturbed = 0
        self.num_words = 0

    def replace_word(self, input_word, threshold=0.5):
    
        self.num_words += 1      
        if self.embed_type == "glove":
            word_embed = self.wv_model.get(input_word)

        elif self.embed_type == "word2vec":
            if input_word in self.wv_model: word_embed = self.wv_model[input_word]
            else: word_embed = None

        if word_embed is None: 
            return input_word

        euclid_dists = np.linalg.norm(self.embedding_matrix[1:] - word_embed, axis=1)

        word_euclid_dict = {word:dist for word, dist in zip(self.vocab_dict, euclid_dists)}

        beta = 0.001
        
        threshold=  round(2/self.epsilon * math.log(((1-beta)*self.vocab_size)/beta),1)

        Lw = [word for word in word_euclid_dict if word_euclid_dict[word] <= threshold]

        f = {word: -word_euclid_dict[word] for word in Lw}

        f["⊥"] = -threshold + 2 * np.log(len(self.vocab_dict)/len(Lw)) / self.epsilon

        f = {word: f[word] + np.random.gumbel(0, 2 / self.epsilon) for word in f}
        
        privatized_word = max(f, key=f.get)

        if privatized_word == "⊥":
            new_word = np.random.choice([word for word in self.vocab_dict if word not in Lw])
            if new_word != input_word: self.num_perturbed += 1
            return new_word
        else:
            if privatized_word != input_word: self.num_perturbed += 1
            return privatized_word

class SynTF:
    def __init__(self, epsilon, sensitivity, vectorizer, data):
        self.epsilon=epsilon
        self.sensitivity=sensitivity
        self.entire_doc = [" ".join([doc for doc in data])]
        self.vectorizer = vectorizer
        self.tfidf_matrix = self.get_tfidf()
        self.words = list(self.tfidf_matrix.index)
        self.syn_dict = { word : self.synonym_extractor(phrase=word) for word in self.words }
        self.syn_scores={ word : self.get_synonym_score(word) for word in list(self.syn_dict.keys()) }

    def get_tfidf(self):
        vectors = self.vectorizer.fit_transform(self.entire_doc)
        tf_idf = pd.DataFrame(vectors.todense())
        tf_idf.columns = self.vectorizer.get_feature_names_out()
        tfidf_matrix = tf_idf.T
        return tfidf_matrix

    def get_synonym_score(self, word):
        score_dict = {}
        for syn in self.syn_dict[word]:
            if syn in self.tfidf_matrix.index: score_dict[syn] = self.tfidf_matrix.loc[word][0]
            else: score_dict[syn] = 0
        return score_dict

    def synonym_extractor(self, phrase):
        synonyms = set()
        for syn in wordnet.synsets(phrase):
            for l in syn.lemmas():
                synonyms.add(l.name())
        return synonyms

    def replace_word(self, word):
        if word not in self.syn_scores: return word
        
        scores = self.syn_scores[word]
        if not scores: return word

        probabilities = [np.exp(self.epsilon * score / (2 * self.sensitivity)) for score in scores.values()]

        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)
        
        return np.random.choice(list(scores.keys()), 1, p=probabilities)[0]

class Mahalanobis:
    def __init__(self, vocab_dict, epsilon, embed_type, wv_model, embedding_matrix, cov_mat, identity_mat, lambd, dim):
        self.vocab_dict = vocab_dict
        self.epsilon = epsilon
        self.wv_model = wv_model
        self.embed_type = embed_type
        self.embedding_matrix = embedding_matrix
        self.cov_mat = cov_mat
        self.identity_mat = identity_mat
        self.lambd = lambd
        self.dim = dim
        self.num_words = 0
        self.num_perturbed = 0

    def get_perturbed_vector(self, word_vec, n):
        noise = np.random.multivariate_normal(np.zeros(n), np.identity(n))
        norm_noise = np.divide(noise, np.linalg.norm(noise))
        Z = np.multiply(np.random.gamma(n, 1/self.epsilon), np.dot(sqrtm(self.lambd*self.cov_mat + (1-self.lambd)*self.identity_mat), norm_noise))
        return word_vec + Z

    def get_nearest(self, vector):
        diff = (self.embedding_matrix - vector)
        most_sim_index = np.argmin(np.linalg.norm(diff, axis=1))

        return most_sim_index

    def get_word_from_index(self, index):
        keys = [k for k, v in self.vocab_dict.items() if v == index]
        if keys:
            return keys[0]
        return None

    def replace_word(self, word):
        if self.embed_type == "glove":
            embedding_vector = self.wv_model.get(word)

        elif self.embed_type == "word2vec":
            if word in self.wv_model: embedding_vector = self.wv_model[word]
            else: embedding_vector = None

        if embedding_vector is not None:
            self.num_words += 1
            perturbed_vector = self.get_perturbed_vector(embedding_vector, self.dim)
            sim_ind = self.get_nearest(perturbed_vector)
            new_word = self.get_word_from_index(sim_ind)
            if new_word is None:  return word
            else:
                self.num_perturbed += 1
                return new_word
        return word