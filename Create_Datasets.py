import pandas as pd
import pickle
from torch.utils.data import random_split


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

    #RETURN SHUFFLED DATASET
    return dataset.sample(frac=1).reset_index(drop=True)


#REMOVES ALL ROWS WHERE THE TEXT VALUE IS EMPTY
def Remove_Empty_Text(dataset):

    empty = []
    for i, row in dataset.iterrows():
        if row['text'].strip() == "": empty.append(i)

    return dataset.drop(empty, axis='index')
   

#WE'RE USING A 60-20-20 SPLIT
def Split_Dataset(dataset):
    return random_split(dataset, [0.6, 0.2, 0.2])

   
#SAVES DATA AS PKL FILE SO WE DON'T HAVE TO KEEP MERGING AND REMOVING EMPTY TEXT
def Save_Merged_Data(dataset):
    with open(f'{PKL_PATH}Merged.pkl', 'wb') as f:
        pickle.dump(dataset, f)


#LOADS DATA FROM PKL FILES
#pickles is funny with versions, this seems to be an okay workaround
def Load_Merged_Data():
    try:
        with open(f'{PKL_PATH}Merged.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        print("Making Merged.pkl from raw CSVs")

        dataset = Merge_Datasets()
        dataset = Remove_Empty_Text(dataset)

        Save_Merged_Data(dataset)
        dataset.to_csv(f"{READABLES}Merged.csv", index=False)

        return dataset



def Create_Dataset():
    
    dataset = Merge_Datasets()
    dataset = Remove_Empty_Text(dataset)
    Save_Merged_Data(dataset)

    #SAVING OUR DATASET AS A NEW CSV SO IT BECOMES HUMAN-READABLE
    dataset.to_csv(f"{READABLES}Merged.csv", index=False)


#RUN THIS IF YOU DON'T HAVE THE PKL FILE FOR THE MERGED DATASET
#Create_Dataset()