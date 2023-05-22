import nltk
nltk.download ("punkt")
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

# function for appending stem words
def get_stem_words(words, ignore_words): #words=[hi,there,thanks,asking]
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)   #stem_words=[hi,there,thank,ask,]
    return stem_words


for intent in intents['intents']:
    
        # Add all words of patterns to list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)             #words=[hi,there,!,!,]           
            word_tags_list.append((pattern_word, intent['tag'])) # wordtaglist=[hi,there,!!,[greeting]]
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  #classes=[greeting,goodbye,thanks,noanswer]
            stem_words = get_stem_words(words, ignore_words) #stem_words=[hi,there,hello]

print(stem_words)
print(word_tags_list[0]) 
print(classes) 
def create_bot_corpus(stem_words, classes):
    stem_words = sorted(list(set(stem_words))) 
    classes = sorted(list(set(classes))) 

    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print(len(stem_words))
print(classes)


       #pattern= hi there,  tag =greeting [0,1,1,1]
       #pattern="see you soon" tag= goodbye [1,0,0,0] thanks[0010]
       
      # classes=['goodbye', 'greeting', 'noanswer', 'thanks']
training_data = []
number_of_tags = len(classes) #4
labels = [0]*number_of_tags #for each tag, greeting [0000]

# Create bag od words and labels_encoding
for word_tags in word_tags_list: #(['Hi', 'there', '!', '!'], 'greeting') (How are you?, "greeting")
        
        bag_of_words = []       
        pattern_words = word_tags[0] #['Hi', 'there', '!', '!'] 
       
        for word in pattern_words:
            index=pattern_words.index(word) #index=0, word= Hi
            word=stemmer.stem(word.lower()) #hi
            pattern_words[index]=word  

        for word in stem_words: # 'anyon', 'are', 'awesom', 'bye', 'chat', 'for', 'goodby', 'hello', 'help', 'hey', 'hi', 'hola', 'how', 'is', 'later', 'me', 'ne', 'next', 'nice', 'see', 'thank', 'that', 'there', 'till', 'time', 'to', 'you']
            if word in pattern_words:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        print(bag_of_words)
        labels_encoding = list(labels) #labels all zeroes initially
        tag = word_tags[1] #save tag []
        tag_index = classes.index(tag)  #go to index of tag  indexposition 1
        labels_encoding[tag_index] = 1  #append 1 at that index [0,1,0,0]
       
        training_data.append([bag_of_words, labels_encoding])

print(training_data[0])

    
