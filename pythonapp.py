#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 14:23:14 2018

@author: amc
"""
import json 
import requests
import time
import urllib

TOKEN = "734444167:AAEjv-xgXoSBVNCg8x5vQvShGgRQOpel4pc"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates():
    url = URL + "getUpdates"
    js = get_json_from_url(url)
    return js


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def send_message(text, chat_id, reply_markup=None):
    text = urllib.parse.quote_plus(text)
    url = URL + "sendMessage?text={}&chat_id={}&parse_mode=Markdown".format(text, chat_id)
    if reply_markup:
        url += "&reply_markup={}".format(reply_markup)
    get_url(url)

items= ['yes', 'no']
def build_keyboard(items):
    keyboard = [[item] for item in items]
    reply_markup = {"keyboard":keyboard, "one_time_keyboard": True}
    return json.dumps(reply_markup)

    
    
 # -------------------- script for A.I. -----------------------#
import numpy
import pandas
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer


ps = SnowballStemmer('english')

def preprocess(text):
            # Stem and remove stopwords
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text = text.split()
            text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
            return ' '.join(text)
        
def update_agent(agent_file):
    dataset = pandas.read_csv(agent_file, encoding='ISO-8859-1')
    querycorpus = []
    for i in range(0, len(dataset)):
        query = re.sub('[^a-zA-Z]', ' ', dataset['Question'][i])
        query = query.lower()
        query = query.split()
        query = [ps.stem(word) for word in query if not word in set(stopwords.words('english'))]
        query = ' '.join(query)
        querycorpus.append(query)       
    # Creating the Bag of Words model with TFIDF and calc cosine_similarity
    vectorizer = CountVectorizer(decode_error="replace")
    vec_train = vectorizer.fit_transform(querycorpus) #this is needed to get the attribute vocabulary_
    training_vocabulary = vectorizer.vocabulary_
    transformer = TfidfTransformer()
    trainingvoc_vectorizer = CountVectorizer(decode_error="replace", vocabulary=training_vocabulary)
    return TfidfVectorizer().fit_transform(querycorpus), transformer, trainingvoc_vectorizer, dataset

tfidf_querycorpus_heather, transformer_heather, trainingvoc_vectorizer_heather, dataset_heather  = update_agent("qna_Heather.csv")
tfidf_querycorpus_joe, transformer_joe, trainingvoc_vectorizer_joe, dataset_joe = update_agent("qna_Joe.csv")

def answer(newquery, k, agent):
    
    if agent=="Joe":
        tfidf_querycorpus = tfidf_querycorpus_joe
        transformer = transformer_joe
        trainingvoc_vectorizer = trainingvoc_vectorizer_joe
        dataset = dataset_joe
    elif agent=="Heather":
        tfidf_querycorpus = tfidf_querycorpus_heather
        transformer = transformer_heather
        trainingvoc_vectorizer = trainingvoc_vectorizer_heather
        dataset = dataset_heather

    tfidf_newquery = transformer.fit_transform(trainingvoc_vectorizer.fit_transform(numpy.array([preprocess(newquery)]))) 
    cosine_similarities = cosine_similarity(tfidf_newquery, tfidf_querycorpus)
    related_docs_indices = (-cosine_similarities[0]).argsort()
    sorted_freq = cosine_similarities[0][related_docs_indices]
    
    #note for this distance the problem we had befor with inf, we have now with 0. Again we decide
    #to make the prediction a bit random. This could be adjusted to remove any 0 distance and
    #pick the only ones left if any, and if none predict 1.
    
    if sum(sorted_freq)==0:
        return "Sorry, I find difficult understanding this question, I'm a poor both only few hours young :("
    
    elif sorted_freq[k-1]!=sorted_freq[k] or sorted_freq[k-1]==sorted_freq[k]==0:
        selected = related_docs_indices[:k]
       
        return dataset.iloc[selected[0]]['Answer']
        #print("\n Cosine Similarities:", sorted_freq, "\n")
        #return dataset.iloc[selected,:k]
    
    else:
        indeces = numpy.where(numpy.roll(sorted_freq,1)!=sorted_freq)
        selected = related_docs_indices[:indeces[0][indeces[0]>=k][0]]
    
        return dataset.iloc[selected[0]]['Answer']
        #print("\n Cosine Similarities:", sorted_freq, "\n")
        #return dataset.iloc[selected,:k]
        
        
        
        
# ---------------------------- launch program -----------------------------#    

def main():
    last_textchat = (None, None)
    selectedagent="Joe" #if I put none, then other var's are undefined in the answer() function
    while True:
        query, chat = get_last_chat_id_and_text(get_updates())
        if (query, chat) != last_textchat:
            if query == "Help me choose a whole life endowment":
                keyboard = build_keyboard(["Heather", "Joe"])
                send_message("Sure, let's book an appointment to go through the details.", chat)
                send_message("OK, I am available on Friday at 3 PM or Monday from 9 AM to 2 PM. When do you prefer to come over?", chat, keyboard)
            elif query == "Heather":
                send_message("Hello Michelle, I'm Heather A.I., ask me anything (related to insurance, please!)", chat)
                selectedagent = "Heather"
            elif query == "Joe":
                send_message("Hello, I'm Joe A.I., ask me anything (related to insurance, please!)", chat)
                selectedagent = "Joe"
            elif query == "/update_Heather":
                tfidf_querycorpus_heather, transformer_heather, trainingvoc_vectorizer_heather, dataset_heather  = update_agent("qna_Heather.csv")
                selectedagent = "Heather"
                send_message("Heather A.I. is up to date! Keep firing questions ;)", chat)
            elif query == "/update_Joe":
                tfidf_querycorpus_joe, transformer_joe, trainingvoc_vectorizer_joe, dataset_joe = update_agent("qna_Joe.csv")
                selectedagent = "Joe"
                send_message("Joe A.I. is up to date! Keep firing questions ;)", chat)
            else:
                text = answer(newquery=get_last_chat_id_and_text(get_updates())[0], k=5, agent=selectedagent)
                send_message(text, chat)
            last_textchat = (query, chat)
        time.sleep(0.5)


if __name__ == '__main__':
    main()



#answer_working(newquery=get_last_chat_id_and_text(get_updates())[0], k=5)
