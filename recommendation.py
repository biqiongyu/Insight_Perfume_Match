import pandas as pd
import numpy as np
import pickle
from textwrap import wrap
import re
import requests
from smart_open import open
from smart_open import smart_open  # before

#NLP tools
import nltk
#nltk.download('vader_lexicon')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

#display features
from IPython.display import display
from ipywidgets import widgets
from IPython.display import clear_output

from skimage import io
import matplotlib.pyplot as plt

from urllib.request import urlopen
from io import BytesIO
from PIL import Image


## define a new class
class Fragrance_Retrieve_Model():

    def __init__(self):
        self.dv = Doc2Vec.load("./models/doc2vec_model")
        self.doctovec_feature_matrix = pickle.load(open("models/doctovec_embeddings.pkl","rb" ))
        self.df = df = pd.read_csv("data/fragrance_data.csv")
        self.hal = sia()


    @staticmethod
    def preprocessText(text):
            
        stops = set(stopwords.words("english"))
        stemmer = SnowballStemmer('english')

        text = str(text).lower() # convert text to lower-case
        text = word_tokenize(text) # remove repeated characters (helloooooooo into hello)    
     
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words) # word stemmer
    
        tokenizer = RegexpTokenizer(r'\w+') #tokenize
        text = tokenizer.tokenize(text)
    
        stop_words = [word for word in text if word not in stops]
        text = " ".join(stop_words)
   
        return text

    def get_message_sentiment(self, message):
        sentences = re.split('\.|\but',message)
        sentences = [x for x in sentences if x != ""]
        love_message = ""
        hate_message = ""
        for s in sentences:
            sentiment_scores = self.hal.polarity_scores(s)
            if sentiment_scores['neg'] > 0:
                hate_message = hate_message + s
            else:
                love_message = love_message + s
        
        return love_message, hate_message


    def preprocess_message(self, message):
        message = self.preprocessText(message)
        return message


    def get_message_doctovec_embedding_vector(self, message):
        message_array = self.dv.infer_vector(doc_words=message.split(" "), epochs=200)
        message_array = message_array.reshape(1, -1)
        return message_array


    @staticmethod
    def get_similarity_scores(message_array, embeddings):
        cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,
                                                           Y=message_array,
                                                           dense_output=True))
        cosine_sim_matrix.set_index(embeddings.index, inplace=True)
        cosine_sim_matrix.columns = ["cosine_similarity"]
        return cosine_sim_matrix


    def get_ensemble_similarity_scores(self, message):
        message = self.preprocess_message(message)
      #  bow_message_array = self.get_message_tfidf_embedding_vector(message)
        semantic_message_array = self.get_message_doctovec_embedding_vector(message)

       # bow_similarity = self.get_similarity_scores(bow_message_array, self.svd_feature_matrix)
        semantic_similarity = self.get_similarity_scores(semantic_message_array, self.doctovec_feature_matrix)

      #  ensemble_similarity = pd.merge(semantic_similarity, bow_similarity, left_index=True, right_index=True)
        semantic_similarity.columns = ["semantic_similarity"]
      #  ensemble_similarity['ensemble_similarity'] = (ensemble_similarity["semantic_similarity"] + ensemble_similarity["bow_similarity"])/2
        semantic_similarity.sort_values(by="semantic_similarity", ascending=False, inplace=True)
        return semantic_similarity


    def get_dissimilarity_scores(self, message):
        message = self.preprocess_message(message)
        #bow_message_array = self.get_message_tfidf_embedding_vector(message)
        semantic_message_array = self.get_message_doctovec_embedding_vector(message)

        dissimilarity = self.get_similarity_scores(semantic_message_array, self.doctovec_feature_matrix)
        dissimilarity.columns = ["dissimilarity"]
        dissimilarity.sort_values(by="dissimilarity", ascending=False, inplace=True)
        return dissimilarity


    def query_similar_perfumes(self, message, n):

        love_message, hate_message = self.get_message_sentiment(message)

        similar_perfumes = self.get_ensemble_similarity_scores(love_message)
        dissimilar_perfumes = self.get_dissimilarity_scores(hate_message)
        dissimilar_perfumes = dissimilar_perfumes.query('dissimilarity > .3')
        similar_perfumes = similar_perfumes.drop(dissimilar_perfumes.index)

        return similar_perfumes.head(n)

    def view_recommendations(self, recs,n):
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(15,10))
        ax = axes.ravel()

        for i in range(len(recs)):
            single_title = recs.index.tolist()[i]
            single_perfume = self.df.query('name==@single_title')
            name = single_perfume.name.values[0]
            description = single_perfume.description.values[0]
            #title = "{} \n Description: {}".format(name, description)
            title = "{}".format(name)

            perfume_image = single_perfume.product_image_url.values[0]
            
            headers = {'User-Agent': 'My User Agent 1.0',
                       'From': 'bqyuyu@gmail.com'}  # This is another valid field
            response = requests.get(perfume_image, headers=headers)
            if response.status_code == 200:
                with open("images/sample{}.jpg".format(i), 'wb') as f:
                    f.write(response.content)
        
            image = io.imread("images/sample{}.jpg".format(i))
           
            ax[i].imshow(image)
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
            ax[i].set_title("\n".join(wrap(title, 20)))
            ax[i].axis('off')

        plt.show()
