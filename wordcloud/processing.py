import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

plt.style.use('ggplot')
import matplotlib.font_manager as fm
import numpy as np
#embedding EN
from gensim.models import KeyedVectors, word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
#embedding TH
from pythainlp import word_vector


def embed_w2v(word_counts):

    words = word_counts.keys()
    model = word_vector.get_model()
    thai2dict = {}
    for word in model.index2word:
        thai2dict[word] = model[word]
    thai2vec = pd.DataFrame.from_dict(thai2dict,orient='index')
    wn_emb = thai2vec.loc[words]
    wn_emb.to_csv('tmp_emb.txt', header=None, index=True, sep=' ', mode='a')
    glove_file = datapath('tmp_emb.txt')
    tmp_file = get_tmpfile("tmp_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    w2vmodel = KeyedVectors.load_word2vec_format(tmp_file)

    return w2vmodel


def plot_TSNE(model,labels=None, filename=None, axis_lims=None):

    tokens = []
    if labels == None:
      labels = []
      for word in model.wv.vocab:
          tokens.append(model[word])
          labels.append(word)
    else:
      for word in labels:
          tokens.append(model[word])

    tsne_model = TSNE(n_components=2, init='pca', n_iter=2250,perplexity=7,early_exaggeration = 12,
                      random_state=26,learning_rate=210)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []                
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for value in new_values:
        if value[0] < min_x:
          min_x = value[0]
        if value[0] > max_x:
          max_x = value[0]
        if value[1] < min_y:
          min_y = value[1]
        if value[1] > max_y:
          max_y = value[1]
          
    if min_x <= 0:
        x_fab = math.fabs(min_x)

    if min_y <= 0:
        y_fab = math.fabs(min_y)

    for value in new_values:
        x.append(value[0] + x_fab)
        y.append(value[1] + y_fab)

    # plt.figure(figsize=(16, 16)) 
    dic = {}
    # prop = fm.FontProperties(fname=f'/content/ChulaCharasNewReg.ttf',size=20)
    for i in range(len(x)):
        dic[labels[i]] = (x[i],y[i])
        # plt.scatter(x[i],y[i])
        # plt.annotate(labels[i],
        #              fontproperties=prop,
        #              xy=(x[i], y[i]),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom',)
    # plt.show()
    return dic


    
