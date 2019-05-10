from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from read_article import read_article
from sentence_similarity import sentence_similarity
from similarity_matrix import build_similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    
    sentences =  read_article(file_name)

    
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    print(" \n")    
    print("Indexes of top ranked_sentence order are ")
    print(" \n")
    print( ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))
    print(" \n")
    
    print("Summarize Text: \n")
    print(". ".join(summarize_text))
