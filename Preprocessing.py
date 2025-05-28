from Create_Datasets import Load_Merged_Data, Split_Dataset, PKL_PATH, READABLES
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import re


SPECIAL_TOKENS = ['<pad>', '<eos>', '<unk>']
PAD_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2

#HYPERPARMETER TUNING
MAX_ARTICLE_LEN = 256
MIN_VOCAB_FREQ = 3


def Get_Labels(datasets):

    labels = []
    for dataset in datasets:
        labels.append(dataset['label'].values)

    return labels


def Preprocess_Text(text):

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
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
        processed_words = processed_words + [ SPECIAL_TOKENS[EOS_INDEX] ]
        
        for word in processed_words:
            processed_text.append(word)
 
    return processed_text


def Build_Vocab(tokenized_text):

    #A VOCABULARY FOR WORDS THAT APPEAR AT LEAST 3 TIMES
    vocab = {token: 0 for token in SPECIAL_TOKENS}

    #ADD ALL WORDS
    for text in tokenized_text:
        for token in text:
            if token not in vocab:
                vocab[token] = 1
            else: vocab[token] = vocab.get(token) + 1

    #REMOVE WORDS THAT DON'T APPEAR OFTEN
    vocab = [ token for token, i in vocab.items() if token in SPECIAL_TOKENS or i >= MIN_VOCAB_FREQ ]

    #RETURN AN ENUMERATED VOCAB
    return { vocab[i] : i for i in range( len(vocab) ) }


def Add_Padding(processed_data):

    for dataset in processed_data:
        for article in dataset:

            #TRUNCATE ARTICLE TO MAX LENGTH IF NECESSARY
            del article[MAX_ARTICLE_LEN:]
            n = MAX_ARTICLE_LEN - len(article)

            for i in range(n):
                article.append(SPECIAL_TOKENS[PAD_INDEX])

    return processed_data


#ENCODES THE TOKENS IN THE DICTIONARY AS NUMBERS
    #THE DICTIONARY IS ENUMERATED SO RETURN THE ENUMERATION
    #IF THE TOKEN IS NOT IN THE DICTIONARY, USE THE UNKNOWN TOKEN'S NUMBER
def Encode_Data(processed_data, vocab):

    encodings = []

    for dataset in processed_data:
        encoded_dataset = []

        for article in dataset:
            encoded_article = []

            for token in article:
                encoded_article.append( vocab.get(token, UNK_INDEX) )

            encoded_dataset.append(encoded_article)
        encodings.append(encoded_dataset)

    return encodings


#SAVES DATA AS PKL FILE SO WE DON'T HAVE TO KEEP REDOING ENCODINGS AND VOCABS
def Save_Data(encoded_data, labels, vocab):

    with open(f'{PKL_PATH}vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(f'{PKL_PATH}labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open(f'{PKL_PATH}encoded_data.pkl', 'wb') as f:
        pickle.dump(encoded_data, f)


def Save_Get_Untruncated(untruncated_data):

    if untruncated_data:
        with open(f'{PKL_PATH}untruncated_data.pkl', 'wb') as f:
            pickle.dump(untruncated_data, f)

    else:
        with open(f'{PKL_PATH}untruncated_data.pkl', 'rb') as f:
            untruncated_data = pickle.load(f)
        return untruncated_data


#LOADS DATA FROM PKL FILES
def Load_Data():

    with open(f'{PKL_PATH}encoded_data.pkl', 'rb') as f:
        encoded_data = pickle.load(f)
    with open(f'{PKL_PATH}labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(f'{PKL_PATH}vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    return encoded_data, labels, vocab


def Preprocess_Data():

    #GET DOWNLOADS
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    
    #LOAD AND SPLIT DATASET
    dataset = Load_Merged_Data()
    training_data, validation_data, test_data = Split_Dataset(dataset)

    #SAVE THE LABELS FOR EACH DATASET
    labels = Get_Labels([ training_data, validation_data, test_data ])

    #PROCESS THE DATASETS
    processed_training = training_data['text'].apply(Preprocess_Text)
    processed_validation = validation_data['text'].apply(Preprocess_Text)
    processed_test = test_data['text'].apply(Preprocess_Text)

    processed_data = [processed_training, processed_validation, processed_test]
    untruncated_data = [ article for dataset in processed_data for article in dataset ]
    Save_Get_Untruncated(untruncated_data)

    #BUILD VOCAB ON TRAINING SET
    vocab = Build_Vocab(processed_training)

    #PAD AND ENCODE DATA
    processed_data = Add_Padding(processed_data)
    encoded_data = Encode_Data(processed_data, vocab)

    Save_Data(encoded_data, labels, vocab)


#CREATES OUTPUT FILES TO VIEW DATA AND VOCAB. GOOD LUCK OPENING IT - IT'S HUGE
def Create_Readable_Text(unprocessed, encoding, labels, vocab):

    with open(f"{READABLES}unprocessed.txt", "w") as f:
        print(unprocessed, file=f)
    with open(f"{READABLES}encoded_data.txt", "w") as f:
        print(encoding, file=f)
    with open(f"{READABLES}labels.txt", "w") as f:
        print(labels, file=f)
    with open(f"{READABLES}vocab.txt", "w") as f:
        print(vocab, file=f)


#RUN THIS IF YOU DON'T HAVE THE PKL FILES FOR ENCODINGS AND VOCAB
#Preprocess_Data()