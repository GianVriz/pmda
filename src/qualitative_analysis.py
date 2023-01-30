import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def barplot_topwords(topic, beta, vocab, topwords=10):
    topK = np.argsort(-1 * beta[topic,])[0:topwords]
    data = {'word': [str(vocab[i]) for i in list(topK)], 'beta': list(beta[topic,topK])}
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize = (10,6))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
        
    sns.barplot(x='beta', y='word', data=df, orient='h', edgecolor='black', color='steelblue')
    plt.title('Top ' + str(topwords) + ' words for topic ' + str(topic), fontsize = 18)
    plt.xlabel('Beta', fontsize=18)
    plt.ylabel('Word', fontsize=18)
    plt.show
    

def visualize_documents(theta, doc):
    data = {'topic': list(range(theta.shape[1])), 'distrib': list(theta[doc,])}
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize = (10,6))
    sns.barplot(x='topic', y='distrib', data=df, color='steelblue')
    plt.title('Topic distribution of document '+ str(doc), fontsize=18)
    plt.xlabel('topic', fontsize=18)
    plt.ylabel('weight', fontsize=18)
    plt.show()
    
    
def visualize_topics(alpha, colours, annotations=[0,0]):
    topics = alpha.shape[0]
    tsne_model = TSNE(n_components=2, verbose=0, random_state=0, angle=.99, init='pca')
    tsne_alpha = tsne_model.fit_transform(alpha)
    indexes = list(range(topics))
    fig, ax = plt.subplots(figsize=(12,6))
    plt.rcParams['font.size'] = '12'
    ax.scatter(x=tsne_alpha[:, 0], y=tsne_alpha[:, 1], c=colours, marker='D', label='Topics')
    for i in list(range(topics)):
        ax.annotate(i, (tsne_alpha[i,0] + annotations[0], tsne_alpha[i,1]  + annotations[1]))
    plt.title('TSNE dimensionality reduction', fontsize = 16)
    plt.show()
    

def visualize_top_words(words, beta, rho, colours, alt_colour='#888888', topwords=10):
    topics = beta.shape[0]
    # creating dataframe for interactive plot
    topK = np.argsort(-1 * beta, axis=1)[:, 0:topwords]
    topKvect = np.concatenate(topK)
    indexes = list(topKvect)
    tsne_topK = words[indexes,]
    colours2 = list()
    for i in range(topics):
        colours2.extend([colours[i]] * topwords)
    # creating dataframe from arrays
    df = pd.DataFrame({'X-axis': np.array(tsne_topK[:, 0]),
                       'Y-axis': np.array(tsne_topK[:, 1]),
                       'Colours': np.array(colours2)},
                      index = range(tsne_topK.shape[0]))
    df['Topic'] = pd.factorize(df['Colours'])[0]
    # interactive plot
    selection = alt.selection_multi(fields=['Topic'], toggle='true')

    chart = alt.Chart(df).mark_circle(size = 50).encode(
        x = alt.X('X-axis', axis=alt.Axis(title = 'X-axis')),
        y = alt.Y('Y-axis', axis=alt.Axis(title = 'Y-axis')),
        color = alt.condition(selection, alt.Color('Topic:N', 
                                                   legend=None, 
                                                   scale=alt.Scale(domain = list(df['Topic']), range=colours)),
                              alt.value(alt_colour))
    ).properties(title="TSNE dimensionality reduction", width=800, height = 350).add_selection(selection).interactive()

    legend = alt.Chart(df).mark_point(filled = True, size = 200).encode(
        y = alt.Y('Topic', axis = alt.Axis(orient = 'right')),
        color = alt.condition(selection, alt.Color('Topic:N',
                                                 legend = None,
                                                 scale = alt.Scale(domain = list(df['Topic']), range = colours)),
                            alt.value(alt_colour))
    ).add_selection(selection)

    background = (alt.Chart(df).mark_rect(color='white', opacity=0.09))

    return background + chart | legend