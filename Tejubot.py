try:
    import random
    import json
    import pickle
    import numpy as np
    import nltk

    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import load_model

    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intence.json').read())
    #Opening the json file for read the data

    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    #There'r two files words&classes are made by traninig phase
    #We are opening these files and read there data for tejubot processing
    model = load_model('Tejubot model.model')

    ''' Now again we got the data but in str mode we hve to canvert and
        clean the data [Hindi me bole to kaat-chant karne ka taaki apn 
        usko apne hisaab se Istemaal kar paanye].If you d't understand
        please learn Hindi it's sach a Amezing Language.
    '''
    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words
    
    '''Now we hve to create bages for our data.who are divided data into sentences
       & Meaning ful neumeric values.
    '''

    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    '''We made a prediction class for predict what thw user 
       said and try to predict Error's
    '''

    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    ''' Nw we hav prediction but we need to make a perfact prediction
        4 that we juge the problility of prediction. 
    '''

    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probablity': str(r[1])})
    return return_list

    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['resposes'])
                break
        return result
    
    print("Go! TejuBot is running!")
    
    while True:
        message = input("")
        ints = predict_class(message)
        res = get_response(ints, intens)
        print(res)

#This code 4 Error handling
except Exception as e:
    print(f"Ye Error h bhai{e}")

else:
    print("The Program is successfull ")

finally:
    print("Ye boi")