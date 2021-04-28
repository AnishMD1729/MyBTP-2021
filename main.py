from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
from Autoencoder import *
from datetime import datetime
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
import pkg_resources
from symspellpy import SymSpell, Verbosity
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import easydict

import warnings
warnings.filterwarnings('ignore', category=Warning)


def preprocess(docs, samp_size=None):

    n_docs = len(docs)
    sentences = []
    token_lists = []
    idx_in = []

    samp = np.random.choice(n_docs, samp_size)

    for i, j in enumerate(samp):

        docs[j] = re.sub(r'([a-z])([A-Z])', r'\1\. \2', docs[j]) docs[j] = docs[j].lower()
        docs[j] = re.sub(r'&gt|&lt', ' ', docs[j])
        docs[j] = re.sub(r'([a-z])\1{2,}', r'\1', docs[j])
        docs[j] = re.sub(r'([\W+])\1{1,}', r'\1', docs[j])
        docs[j] = re.sub(r'\*|\W\*|\*\W', '. ', docs[j])
        docs[j] = re.sub(r'\W+?\.', '.', docs[j])
        docs[j] = re.sub(r' ing ', ' ', docs[j])
        docs[j] = re.sub(r'(.{2,}?)\1{1,}', r'\1', docs[j])
        sentence = docs[j].strip()

        en_stop = get_stop_words('en')
        en_stop.append('model')
        en_stop.append('algorithm')
        en_stop.append('data')
        w_list = word_tokenize(sentence)
        w_list = [word for word in w_list if word.isalpha()]
        w_list = [word for (word, pos) in nltk.pos_tag(
            w_list) if pos[:2] == 'NN']
        w_list = [p_stemmer.stem(word) for word in w_list]
        w_list = [word for word in w_list if word not in en_stop]
        token_list = w_list

        if token_list:
            idx_in.append(j)
            sentences.append(sentence)
            token_lists.append(token_list)
        print('{} %'.format(str(np.round((i + 1) / len(samp)*100, 2))), end='\r')

    return sentences, token_lists, idx_in


class Topic_Model:

    def vectorize(self, sentences, token_lists, method=None):

        if method is None:
            method = self.method
        self.dictionary = corpora.Dictionary(token_lists)
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'LDA':
            if not self.ldamodel:
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)

            n_doc = len(corpus)
            vec_lda = np.zeros((n_doc, k))
            for i in range(n_doc):
                for topic, prob in model.get_document_topics(corpus[i]):
                    vec_lda[i, topic] = prob

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            return vec

        elif method == 'BERT':

            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            vec = np.array(model.encode(sentences))
            return vec

        else:
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='BERT')
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
            self.vec['LDA_BERT_FULL'] = vec_ldabert
            if not self.AE:
                self.AE = Autoencoder()
                print('Fitting Autoencoder ...')
                self.AE.fit(vec_ldabert)
                print('Fitting Autoencoder Done!')
            vec = self.AE.encoder.predict(vec_ldabert)
            return vec

    def fit(self, sentences, token_lists, method=None, m_clustering=None):

        if m_clustering is None:
            m_clustering = KMeans

        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            self.corpus = [self.dictionary.doc2bow(
                text) for text in token_lists]

        self.cluster_model = m_clustering(self.k)
        self.vec[method] = self.vectorize(sentences, token_lists, method)
        self.cluster_model.fit(self.vec[method])

    def predict(self, sentences, token_lists, out_of_sample=None):

        out_of_sample = out_of_sample is not None

        if out_of_sample:
            corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.method != 'LDA':
                vec = self.vectorize(sentences, token_lists)
                print(vec)
        else:
            corpus = self.corpus
            vec = self.vec.get(self.method, None)

        if self.method == "LDA":
            lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
                                                     key=lambda x: x[1], reverse=True)[0][0],
                                    corpus)))
        else:
            lbs = self.cluster_model.predict(vec)
        return lbs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    args = easydict.EasyDict({
        '--fpath': 'https://drive.google.com/file/d/1eRhbP2JwxMg36A5YmmB3BfXJugzwK9a7/view?usp=sharing',
        '--ntopic': 10,
        '--method': 'LDA_BERT',
        '--samp_size': 10000
    })
    x = 10000
    data = pd.read_csv(str('test7.csv'))
    data = data.fillna('')
    # data.head(5)

    rws = data.review
    sentences, token_lists, idx_in = preprocess(rws, 60)
    tm = Topic_Model(k=int(3), method=str('LDA_BERT'))
    tm.fit(sentences, token_lists)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))
    visualize(tm)
    for i in range(tm.k):
        get_wordcloud(tm, token_lists, i)
