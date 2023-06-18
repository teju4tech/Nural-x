
try:
    import random #To do random operation
    import json #To handle json file
    import pickle #To strelestion
    import numpy as np 
    import nltk #It's importing Natural language tool kit 
    
    '''Exmple: work,working,worked,works all meaning same'''
    from nltk.stem import WordNetLemmatizer #It's sink word if their meaning same
    
    '''using tensorflow for models and languaage correctizer'''
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Activation,Dropout
    from tensorflow.keras.optimizers import SGD
    
    
    lemmatizer = WordNetLemmatizer
    
    intents = json.loads(open('intence.json').read())
    #Open json file and read
    words = []
    classes = []
    documents = []
    ignore_latters = ['?','!','.',',']
    
    for intent in intents['intens']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern) #tokenize = cut the word at individuals ['I','LOVE','U']
            words.extend(word_list)
            documents.append((word_list),intent['tag'])
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    
    # print(documents)
    words = [lemmatizer.lemmatizer(word) for word in words if word not in ignore_latters]
    words = sorted(set(words))
    # print(words)
    classes = sorted(set(classes))
    # We cread a pk1 file for words
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    '''Now we have a bunch of words, but Neurl-net can not
       accepts words it's needed numeric values, so we have 
       canvert our data to neumeric values.
    '''
    # -------Looop--------
    training = []
    output_empty = [0] * len(classes)
    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatizer(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(documents[1])] = 1
        training.append([bag, output_row])
        #when we run the loop all document data is gone to training[]

    
    # Last preprocessing to create Nueral-Network
    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:,0])
    train_y = list(training[:,1])

    # Now starts to Bulid Neuro-NEt
    #this number was how many neurons in our network defined
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    
    sgd = SGD(lr=0.01,decay=le-6, momentum=0.9,nesterov=True)
    # lr is learining rate of model
    model.compile(loss='categorical crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('Tejubot model.h5',hist)
    print("Traninig Complete !")


except Exception as e:
    print(f"Ye Error h bhai:{e}")

else:
    print("Programm Run successful")

finally:
    print("Le bhiaye OmFo")


