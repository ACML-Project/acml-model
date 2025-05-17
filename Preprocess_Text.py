import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import torch.nn as nn


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

MAX_ARTICLE_LEN = 512
EMBEDDING_DIM = 128
PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3


def Preprocess_Text(text):

    processed_text = []

    #TOKENIZE: SPLIT INTO SENTENCES
    sentences = sent_tokenize(text)

    for sentence in sentences:

        #REMOVE SPECIAL CHARACTERS, DEALS WITH WHITESPACE AND MAKES SENTENCE LOWERCASE
        processed_sentence = re.sub(r'[^a-zA-Z\s]', ' ', sentence.lower()).strip()

        #TOKENIZE: SPLIT INTO WORDS
        words = word_tokenize(processed_sentence)

        #REMOVE STOPWORDS AND LEMMATIZE
        processed_words = []

        for word in words:
            if word not in stop_words:
               processed_words.append(lemmatizer.lemmatize(word))

        #ADD START- AND END-OF-SENTENCE MARKERS
        processed_words = [ special_tokens[SOS_INDEX] ] + processed_words + [ special_tokens[EOS_INDEX] ]
        
        for word in processed_words:
            processed_text.append(word)
 
    #TRUNCATES TEXT TO MAX LENGTH
    del processed_text[MAX_ARTICLE_LEN:]
    
    return processed_text


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
    vocab = [ token for token, i in vocab.items() if token in special_tokens or i > 2 ]

    #RETURN AN ENUMERATED VOCAB
    return { vocab[i] : i for i in range( len(vocab) ) }


def Add_Padding(processed_data):

    for article in processed_data:
        n = MAX_ARTICLE_LEN - len(article)

        for i in range(n):
            article.append(special_tokens[PAD_INDEX])

    return processed_data


dataset = pd.read_csv('Training.csv')
processed_data = dataset['text'].apply(Preprocess_Text)
vocab = Build_Vocab(processed_data)
processed_data = Add_Padding(processed_data)

embedding = nn.Embedding(
    num_embeddings = len(vocab),
    embedding_dim = EMBEDDING_DIM,
    padding_idx = PAD_INDEX
)

#output_file = open("processed_data.txt", "w")
#print(*processed_data, file=output_file)
