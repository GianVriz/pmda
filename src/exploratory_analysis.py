import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

from collections import Counter
from PIL import Image
from wordcloud import WordCloud
   

def most_used_words(docs_bow, num_best_words=15):
    docs_bowf = [item for sublist in docs_bow for item in sublist]
    docs_bowf_count = Counter(docs_bowf)

    tot_dword = dict(sorted(docs_bowf_count.items(), key = lambda item: item[1]))

    best_tot_dwords = dict(sorted(tot_dword.items(), key = lambda item: item[1], reverse=True)[:num_best_words])

    ind = np.arange(len(best_tot_dwords))
    palette = sns.color_palette('husl', len(best_tot_dwords))
    best_tot_dwords_2 = dict(sorted(best_tot_dwords.items(), key=lambda item: item[1]))

    fig, ax = plt.subplots(figsize = (12,6))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)

    plt.barh(ind, list(best_tot_dwords_2.values()), align='center', color='steelblue', edgecolor='black')
    plt.yticks(ind, list(best_tot_dwords_2.keys()))
    plt.ylabel('Words', fontsize=16)
    plt.xlabel('Count', fontsize=16)
    plt.title('Most used words', fontsize=16)
    plt.show()
    

def top_authors(corpus, num_best_authors=15):
    absolute_frequencies = dict()
    for author in corpus['author']:
        if author in absolute_frequencies.keys():
            absolute_frequencies[author] += 1
        else:
            absolute_frequencies[author] = 1
    
    absolute_frequencies_2 = dict(sorted(absolute_frequencies.items(), key=lambda item:item[1], reverse=True)[:num_best_authors])
    absolute_frequencies_2.pop(np.nan, None) # remove unkown authors.  

    ind = np.arange(len(absolute_frequencies_2))
    absolute_frequencies_3 = dict(sorted(absolute_frequencies_2.items(), key = lambda item: item[1]))

    fig, ax = plt.subplots(figsize = (12,6)) 
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    plt.barh(ind, list(absolute_frequencies_3.values()), align='center', edgecolor='black', color='steelblue')
    plt.yticks(ind, list(absolute_frequencies_3.keys()))
    plt.ylabel('Author', fontsize=16)
    plt.xlabel('Articles', fontsize=16)
    plt.title('Authors with most published articles', fontsize=16)
    plt.show()
    
    
def wordcloud_corpus(docs_bow):
    docs_bowf = [item for sublist in docs_bow for item in sublist]
    docs_bowf_count = Counter(docs_bowf)

    tot_dword = dict(sorted(docs_bowf_count.items(), key = lambda item: item[1]))
    wc = WordCloud(background_color='white',
                   mask=np.array(Image.open('images/guardian.png')),
                   contour_width=30,
                   colormap='Blues',
                   contour_color='dodgerblue').generate_from_frequencies(tot_dword)

    plt.figure(figsize = (12,12))
    plt.imshow(wc, interpolation='bilinear', aspect='auto')
    plt.axis('off')
    plt.show()