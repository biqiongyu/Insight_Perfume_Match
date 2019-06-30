import pandas as pd
import numpy as np
import pickle
import re
import requests

#NLP tools
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

from gensim import corpora, models, matutils
from gensim.test.utils import get_tmpfile

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

## define a new class
class Fragrance_Retrieve_Model():

    def __init__(self):
        self.dv = models.Doc2Vec.load("flaskexample/models/doc2vec_model")
        self.tfidf = models.TfidfModel.load('flaskexample/models/tfidf')
        self.lsi = models.LsiModel.load('flaskexample/models/lsimodel')
        self.dictionary = corpora.Dictionary.load('flaskexample/models/dictionary')
        self.doctovec_feature_matrix = pickle.load(open("flaskexample/models/doctovec_embeddings.pkl","rb" ))
        self.lsi_matrix = pickle.load(open("flaskexample/models/lsi_embeddings.pkl","rb" ))
        self.df = df = pd.read_csv("flaskexample/data/fragrance_data.csv")


    @staticmethod
    def preprocessText(text):
            
        stops = stopwords.words("english")
        #add more stopwords in this particular case
        stops.extend(['love','like','hate','amazing','favorite','dislike',"don't",'awesome','great','good','bad','horrible','excellent'])
        
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


    def preprocess_message(self, message):
        message = self.preprocessText(message)
        return message


    def get_message_doctovec_embedding_vector(self, message):
        message_array = self.dv.infer_vector(doc_words=message.split(" "))
        message_array =  message_array.reshape(-1, 1).T
        return message_array
    
    def get_message_lsi_embedding_vector(self, message):
        test_corpus = [self.dictionary.doc2bow(message.split())]
        test_corpus_tfidf = self.tfidf[test_corpus]
        test_lsi = self.lsi[test_corpus_tfidf]
        test_vector = matutils.corpus2csc(test_lsi)
        message_array = test_vector.toarray().reshape(-1, 1).T
        return message_array

    @staticmethod
    def get_similarity_scores(message_array, embeddings):
        if message_array.size == 0:
            a = np.zeros(shape=(embeddings.shape[0],1))
            cosine_sim_matrix = pd.DataFrame(a,columns=["cosine_similarity"])
            cosine_sim_matrix.set_index(embeddings.index, inplace=True)
        else:
            cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,
                                                           Y=message_array,
                                                           dense_output=False))
            cosine_sim_matrix.set_index(embeddings.index, inplace=True)
            cosine_sim_matrix.columns = ["cosine_similarity"]
        return cosine_sim_matrix


    def get_ensemble_similarity_scores(self, message):
        message = self.preprocess_message(message)
        lsi_message_array = self.get_message_lsi_embedding_vector(message)
        semantic_message_array = self.get_message_doctovec_embedding_vector(message)

        lsi_similarity = self.get_similarity_scores(lsi_message_array, self.lsi_matrix)
        semantic_similarity = self.get_similarity_scores(semantic_message_array, self.doctovec_feature_matrix)

        similarity = pd.merge(semantic_similarity, lsi_similarity, left_index=True, right_index=True)
        similarity.columns = ["semantic_similarity","lsi_similarity"]
        similarity['tot_similarity'] = (similarity["semantic_similarity"]*0.3 + similarity["lsi_similarity"]*0.7)
        similarity.sort_values(by="tot_similarity", ascending=False, inplace=True)
        return similarity


    def get_dissimilarity_scores(self, message):
        if message:
            message = self.preprocess_message(message)
            lsi_message_array = self.get_message_lsi_embedding_vector(message)
            semantic_message_array = self.get_message_doctovec_embedding_vector(message)

            lsi_dissimilarity = self.get_similarity_scores(lsi_message_array,self.lsi_matrix)
            semantic_dissimilarity = self.get_similarity_scores(semantic_message_array, self.doctovec_feature_matrix)

            dissimilarity = pd.merge(semantic_dissimilarity, lsi_dissimilarity, left_index=True, right_index=True)
            dissimilarity.columns = ["semantic_dissimilarity","lsi_dissimilarity"]
            dissimilarity['tot_dissimilarity'] = (dissimilarity["semantic_dissimilarity"]*0.3 + dissimilarity["lsi_dissimilarity"]*0.7)    
            dissimilarity.sort_values(by="tot_dissimilarity", ascending=False, inplace=True)
        else:
            dissimilarity = pd.DataFrame()
            dissimilarity['name'] = self.df['name']
            dissimilarity.set_index('name',inplace=True)
            dissimilarity['tot_dissimilarity'] = 0
        
        return dissimilarity


    def query_similar_perfumes(self, love_message, hate_message):
        if love_message:
            similar_perfumes = self.get_ensemble_similarity_scores(love_message)
            dissimilar_perfumes = self.get_dissimilarity_scores(hate_message)
            dissimilar_perfumes = dissimilar_perfumes.query('tot_dissimilarity > 0.2')
            similar_perfumes = similar_perfumes.drop(dissimilar_perfumes.index)
            similar_perfumes = similar_perfumes.query('tot_similarity > 0.28')
            if similar_perfumes.shape[0]>10:
                 similar_perfumes = similar_perfumes.head(10)

            if similar_perfumes.shape[0]==0:
                return None
            else: 
                return similar_perfumes
        else:
            return None
            
