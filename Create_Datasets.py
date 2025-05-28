import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


#THE FOLDERS WHERE OUR DATASET WILL BE SAVED
PKL_PATH = "Pickled_data/"
READABLES = "Readables/"


#MERGES AND SHUFFLES FAKE AND REAL NEWS CSVS 
def Merge_Datasets():

    fake_news = pd.read_csv('Fake.csv')
    true_news = pd.read_csv('True.csv')

    #ADD LABELS SO WE CAN TELL THEM APART
    fake_news['label'] = 0
    true_news['label'] = 1
    
    dataset = pd.concat([fake_news, true_news], axis=0)
    dataset = dataset.drop_duplicates(subset=['text'])

    #RETURN SHUFFLED DATASET
    return dataset.sample(frac=1).reset_index(drop=True)


#REMOVES ALL ROWS WHERE THE TEXT VALUE IS EMPTY
def Remove_Empty_Text(dataset):

    empty = []
    for i, row in dataset.iterrows():
        if row['text'].strip() == "": empty.append(i)

    return dataset.drop(empty, axis='index').reset_index()
   

#WE'RE USING A 60-20-20 SPLIT
def Split_Dataset(dataset):
    
    training_data, temp_data = train_test_split(dataset, test_size=0.4, random_state=42, stratify=dataset['label'])
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])
    return training_data, validation_data, test_data

   
#SAVES DATA AS PKL FILE SO WE DON'T HAVE TO KEEP MERGING AND REMOVING EMPTY TEXT
def Save_Merged_Data(dataset):
    with open(f'{PKL_PATH}Merged.pkl', 'wb') as f:
        pickle.dump(dataset, f)


#LOADS DATA FROM PKL FILES
def Load_Merged_Data():
    with open(f'{PKL_PATH}Merged.pkl', 'rb') as f:
        merged_dataset = pickle.load(f)
    return merged_dataset


def Create_Dataset():
    
    dataset = Merge_Datasets()
    dataset = Remove_Empty_Text(dataset)
    Save_Merged_Data(dataset)

    #SAVING OUR DATASET AS A NEW CSV SO IT BECOMES HUMAN-READABLE
    dataset.to_csv(f"{READABLES}Merged.csv", index=False)


#RUN THIS IF YOU DON'T HAVE THE PKL FILE FOR THE MERGED DATASET
#Create_Dataset()