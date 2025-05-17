import pandas as pd
import numpy as np


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

    n = len(dataset)

    return np.split(
        dataset.sample(frac=1, random_state=42),
        [int(0.6*n), int(0.8*n)])


dataset = Merge_Datasets()
dataset = Remove_Empty_Text(dataset)
training_data, test_data, validation_data = Split_Dataset(dataset)

#SAVING OUR DATASETS AS NEW CSVS SO WE DON'T HAVE TO MERGE AND EVERY TIME
training_data.to_csv("Training.csv", index=False)
test_data.to_csv("Test.csv", index=False)
validation_data.to_csv("Validation.csv", index=False)
