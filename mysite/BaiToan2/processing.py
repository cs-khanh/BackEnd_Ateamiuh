import os
import nltk
import re
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
#nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
def text_clean(text):
    text = re.sub(r"\n"," ",text)   #remove line breaks
    text = text.lower() #convert to lowercase
    text = re.sub(r"\d+","",text)   #remove digits and currencies
    text = re.sub(r'[\$\d+\d+\$]', "", text)
    text = re.sub(r'[()<>::]', " ", text)
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)   #remove dates
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)
    text = re.sub(r'[^\x00-\x7f]',r' ',text)   #remove non-ascii
    text = re.sub(r'[^\w\s]','',text)   #remove punctuation
    text = re.sub(r'https?:\/\/.*[\r\n]*', ' ', text)   #remove hyperlinks
    text = re.sub(r'\b\w{1,3}\b', '',text) # remove words have length from 1 to 3
    text = re.sub(r'\b\w{11,20}\b', '',text) # remove words have length from 1 to 3
    return text
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def lemmatize_passage(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence
def text_sementic(text, method, rm_stop):
    text = text_clean(text)
    if rm_stop == True:
        filtered_tokens = [word for word in word_tokenize(text) if not word in set(stopwords.words('english'))]
        text = " ".join(filtered_tokens)

    if method == 'L':
        lemmer = WordNetLemmatizer()
        lemm_tokens = [lemmer.lemmatize(word) for word in word_tokenize(text)]
        return " ".join(lemm_tokens)

    #stemming
    if method == 'S':
        porter = PorterStemmer()
        stem_tokens = [porter.stem(word) for word in word_tokenize(text)]
        return " ".join(stem_tokens)
    if method == 'T':
      text = lemmatize_passage(text)

    return text
def Words2Text(words):
    text = " ".join(words)
    return text;
def Text2Words(text):
    words = [word for word in word_tokenize(text)]
    return words
def addFeature(tfidf,process,mssv,numberoftimes):
    response=tfidf.transform([process])
    label=['Mã số ID','count']
    labelEnd=['effort']
    feature_names=np.array(tfidf.get_feature_names_out())
    label=np.append(label,feature_names)
    label=np.append(label,labelEnd)
    feature = response.toarray()[0]
    mssv=[mssv,1]
    value=np.append(mssv,feature)
    value=np.append(value,[1/numberoftimes])
    np.set_printoptions(suppress=True)
    df_test= pd.DataFrame([value], columns=label)
    return df_test