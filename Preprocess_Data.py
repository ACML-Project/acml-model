import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def Get_Dataset():

    fake_news = pd.read_csv('Fake.csv')
    true_news = pd.read_csv('True.csv')

    #ADD LABELS
    fake_news['label'] = 0
    true_news['label'] = 1
    dataset = pd.concat([fake_news, true_news], axis=0)

    #RETURN SHUFFLED DATASET
    return dataset.sample(frac=1).reset_index(drop=True)


def Preprocess_Text(text):

    text = text.lower()

    #REMOVE SPECIAL CHARACTERS
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #TOKENIZE
    words = text.split()

    #REMOVE STOPWORDS AND LEMMATIZE
    processed_words = []

    for word in words:
        if word not in stop_words:
            processed_words.append(lemmatizer.lemmatize(word))
    
    return ' '.join(processed_words)


dataset = Get_Dataset()
dataset['text'] = dataset['text'].apply(Preprocess_Text)

#SAVING OUR DATASET AS A NEW CSV SO WE DON'T HAVE TO PREPROCESS EVERY TIME
dataset.to_csv("Merged.csv", index=False)