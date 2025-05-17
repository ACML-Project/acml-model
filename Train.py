

learning_rate = 1
num_epochs = 20
batch_size = 32
max_len = 128


#INDEPENDENT 
#X = dataset.drop('label', axis=1)

#DEPENDENT
#Y = dataset['label']

#print(len(vocab))
#output_file = open("output.txt", "w")
#print(vocab, file=output_file)
#https://www.youtube.com/watch?v=k3_qIfRogyY
#return ' '.join(processed_words)


#KEEP TRACK OF THE LONGEST ARTICLE
    #global MAX_ARTICLE_LEN
    #len_curr = len(processed_text)
    #if len_curr > MAX_ARTICLE_LEN:
    #    MAX_ARTICLE_LEN = len_curr
