import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import urllib
import requests

def wordcloud(cluster):
    # combining the image with the dataset
    Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))
    # We use the ImageColorGenerator library from Wordcloud 
    # Here we take the color of the image and impose it over our wordcloud
    image_colors = ImageColorGenerator(Mask)
    
    # Now we use the WordCloud function from the wordcloud library 
    wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(cluster)
    
    # Size of the image generated 
    plt.figure(figsize=(10,20))
    
    # Here we recolor the words from the dataset to the image's color
    # recolor just recolors the default colors to the image's blue color
    # interpolation is used to smooth the image generated
    plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")
    
    plt.axis('off')
    plt.show()

def identify_topics(data, desc_matrix, num_clusters):
    km = KMeans(n_clusters=num_clusters)
    km.fit(desc_matrix)
    clusters = km.labels_.tolist()
    text = {'Clean_text': data['Clean_text'].tolist(), 'Cluster': clusters}
    frame = pd.DataFrame(text, index= [clusters])
    print(frame['Cluster'].value_counts())
    
    for cluster in range(num_clusters):
        cluster_words = ' '.join(text for text in frame[frame['Cluster']== cluster]['Clean_text'])
        wordcloud(cluster_words)
