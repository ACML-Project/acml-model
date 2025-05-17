import nltk
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']


def Preprocess_Text(text):

    text = text.lower()

    #REMOVE SPECIAL CHARACTERS
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    #DEALS WITH WHITESPACE
    text = re.sub(r'\s+', ' ', text).strip()

    #TOKENIZE
    words = word_tokenize(text)

    #REMOVE STOPWORDS AND LEMMATIZE
    processed_words = []

    for word in words:
        if word not in stop_words and word not in processed_words:
            processed_words.append(lemmatizer.lemmatize(word))

    #ADD START- AND END-OF-SENTENCE MARKERS
    processed_words = [ special_tokens[1] ] + processed_words + [ special_tokens[2] ]

    return processed_words


def Build_Vocab(tokenized_text):

    #A VOCABULARY FOR WORDS THAT APPEAR AT LEAST 3 TIMES
    vocab = {token: 0 for token in special_tokens}

    #ADD ALL WORDS
    for text in tokenized_text:
        for token in text:
            if token not in vocab:
                vocab[token] = 1
            else: vocab[token] = vocab.get(token) + 1

    #REMOVE WORDS THAT DON'T APPEAR OFTEN
    return [ token for token, i in vocab.items() if token in special_tokens or i > 2 ]


training_data = pd.read_csv('Training.csv')
processed_text = training_data['text'].apply(Preprocess_Text)
vocab = Build_Vocab(processed_text)

#print(len(vocab))
#output_file = open("output.txt", "w")
#print(vocab, file=output_file)
#https://www.youtube.com/watch?v=k3_qIfRogyY
#return ' '.join(processed_words)
