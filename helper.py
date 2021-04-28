from collections import Counter
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import os


def get_coherence(model, token_lists, measure='c_v'):

    if model.method == 'LDA':
        cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    else:
        if k is None:
            k = len(np.unique(model.cluster_model.labels_))
        topics = ['' for _ in range(k)]
        for i, c in enumerate(token_lists):
            topics[model.cluster_model.labels_[i]] += (' ' + ' '.join(c))
        word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
        word_counts = list(map(lambda x: sorted(
            x, key=lambda x: x[1], reverse=True), word_counts))
        topics = list(
            map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))
        cm = CoherenceModel(topics=topics, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    return cm.get_coherence()


def get_silhouette(model):

    if model.method == 'LDA':
        return
    lbs = model.cluster_model.labels_
    vec = model.vec[model.method]
    return silhouette_score(vec, lbs)


def visualize(model):

    if model.method == 'LDA':
        return
    reducer = umap.UMAP()
    vec_umap = reducer.fit_transform(model.vec[model.method])
    n = len(vec_umap)
    counter = Counter(model.cluster_model.labels_)
    for i in range(len(np.unique(model.cluster_model.labels_))):
        plt.plot(vec_umap[:, 0][model.cluster_model.labels_ == i], vec_umap[:, 1][model.cluster_model.labels_ == i], '.', alpha=0.5,
                 label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
    plt.legend()


def get_wordcloud(model, token_lists, topic):

    if model.method == 'LDA':
        return

    lbs = model.cluster_model.labels_
    tokens = ' '.join([' '.join(_)
                       for _ in np.array(token_lists)[lbs == topic]])

    wordcloud = WordCloud(width=800, height=560,
                          background_color='white', collocations=False,
                          min_font_size=10).generate(tokens)

    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
