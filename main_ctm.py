from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
from Autoencoder import *
from datetime import datetime
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import easydict

import warnings
warnings.filterwarnings('ignore', category=Warning)


def preprocess(docs, samp_size=None):

    print('Preprocessing raw texts ...')
    n_docs = len(docs)
    sentences = []
    token_lists = []
    idx_in = []

    samp = np.random.choice(n_docs, samp_size)
    for i, idx in enumerate(samp):
        sentence = preprocess_sent(docs[idx])
        token_list = preprocess_word(sentence)
        if token_list:
            idx_in.append(idx)
            sentences.append(sentence)
            token_lists.append(token_list)
        print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
    print('Preprocessing raw texts. Done!')
    return sentences, token_lists, idx_in


class Topic_Model:

    def vectorize(self, sentences, token_lists, method=None):

        if method is None:
            method = self.method
        self.dictionary = corpora.Dictionary(token_lists)
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'LDA_BERT':
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

        if method != 'LDA':
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            self.vec[method] = self.vectorize(sentences, token_lists, method)
            self.cluster_model.fit(self.vec[method])
            print('Clustering embeddings. Done!')

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
