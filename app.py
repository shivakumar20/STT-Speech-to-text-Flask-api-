import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from flask import Flask, jsonify, make_response, render_template, request, redirect, current_app, abort
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import queue
import sys
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


q = queue.Queue()

#Callback :function
#Used for listening the audio   
def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))
    

    
#Dataset Preparation
df = pd.read_csv ('Corpus_with_augmented_data.csv')

#print (df)

df.isnull().sum()

df = df.dropna(how='any',axis=0)

df.isnull().sum()

sns.countplot(x='Category', data=df)

sns.countplot(x='Category',hue ='Sub-category', data=df)

##### List of Questions to list of words for better visualization
word_list=[]
list_question=list(df["best_of_augmented"])
for sentence in list_question:
    words_sentence=sentence.split()
    for words in words_sentence:
        word_list.append(words)

        

word_list=[word for sentence in list(df["best_of_augmented"]) for word in sentence.split()]
#print(word_list)


#####  frequency of each word and the most common words in the dataset
frequency=Counter(word_list)
#print (frequency)
#print (frequency.most_common(5))

### Remove Punctuations and change all words to lower case

def remove_punctuations(text):    
    words=[word.lower() for word in text.split()] 
    words=[w for word in words for w in re.sub(r'[^\w\s]','',word).split()]    
    return words

df["que_no_punct"]=df["best_of_augmented"].apply(remove_punctuations)
#print (df["que_no_punct"])

def negative_words(words):
      counter=False    
      wordlist=[]    
      negatives=["no","not","cant","cannot","never","less","without","barely","hardly","rarely","no","not","noway","didnt"]
    #for words in wordlist:       
      for i,j in enumerate(words):                           
            if j in negatives and i<len(words)-1:             
                wordlist.append(str(words[i]+'-'+words[i+1]))
                counter=True
            else:
                if counter is False:                
                    wordlist.append(words[i])
                else:
                    counter=False
      return wordlist
    
df["question_negative"]=df["que_no_punct"].apply(negative_words)
#print (df["question_negative"])

### Stemming Words

st=PorterStemmer()
def Stem(text):
    stemmed_words=[st.stem(word) for word in text] 
    return stemmed_words

df["question_stem"]=df["question_negative"].apply(Stem)
#print (df["question_stem"])

### Recreate the sentence
def Recreate(text):
    word=" ".join(text)
    return word

df["modified_questions"]=df["question_stem"].apply(Recreate)
#print (df["modified_questions"])

### Let's change the sentence into a bag of word model

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["modified_questions"]).toarray()
#print(X)
#print(vectorizer.get_feature_names())

def Cleaning_questions(text):
    No_text_punctuation=remove_punctuations(text)
    No_Negative_words=negative_words(No_text_punctuation)
    text_stem=Stem(No_Negative_words)
    final_questions=Recreate(text_stem)
    return final_questions
df["modified_Questions"]=df["best_of_augmented"].apply(Cleaning_questions)
#print (df["modified_Questions"])

###### Extra Tf-idf transformation and DataPipelines

model = Pipeline([('vectoizer', CountVectorizer()),
 ('tfidf', TfidfTransformer())])

X_train = model.fit_transform(df["best_of_augmented"]).toarray()
#print(X_train)

Y=df["Answers"]

#question = " How about join RACE programs?"


clf1 = MultinomialNB().fit(X_train, Y)

li = ["What is the duration of each program?",
"What are the prerequisites to join corporate programs of Race?",
"Maximum number of participants for each program of RACE",
"What about the program delivery modes of RACE programs?",
"Do I have the option of completing the RACE programs online?",
"How do I register for the program of RACE I am interested in?",
"Do you provide learning materials to the program participants of RACE?",
"What kind of infrastructure facilities can I expect at RACE?",
"Am I eligible for any discounts to join RACE programs?",
"What are the tools used for hands-on learning in RACE programs?",
"Is it possible to attend a classroom session before joining the RACE program?",
"What can I expect after the completion of the RACE program?",
"What about the admission cycle of RACE programs?",
"How many seats are available for long-term programs of RACE?",
"Is it possible to pay the fees in instalments for RACE program?",
"Do you offer placement assistance at RACE?",
"Why should I join MS in Business Analytics program of RACE?",
"What is the scope of business analytics?",
"When do the admissions for the MS in Business Analytics start at RACE?",
"Prerequisites to join MS in Business Analytics of RACE",
"Is industry experience mandatory for MS in Business Analytics of RACE?",
"How can I learn more about the MS in Business Analytics program of RACE?",
"What is the duration of the MS in Business Analytics program of RACE?",
"Is the MS in Business Analytics program of RACE recognized?",
"How do you promote experiential learning in MS in Business Analytics of RACE?",
"Do you have certification programs in analytics of RACE?",
"How an M. Tech. in Artificial Intelligence program of RACE helps me?",
"Who can apply to M. Tech in AI of RACE?",
"Who can apply to M. Sc. in AI?",
"What subjects do I study in Artificial Intelligence of RACE?",
"How does RACE deliver artificial intelligence programs?",
"Are there any artificial intelligence courses offered online by RACE?",
"What is the demand for artificial intelligence professionals?",
"What is the duration of RACE’s artificial intelligence programs?",
"When does the new batches for Artificial intelligence programs start at RACE?",
"Is industry experience mandatory for joining AI programs of RACE?",
"Is M. Tech in Artificial intelligence program of RACE accredited?",
"How can I learn more about the Artificial intelligence programs of RACE?",
"How do RACE promotes experiential learning in Artificial intelligence?",
"Do you have certification programs in artificial intelligence at RACE?",
"Why should I learn M. Tech in Cybersecurity at RACE?",
"Why should I learn M.Sc. in Cybersecurity at RACE?",
"Why should I learn PGD in Cybersecurity of RACE?",
"Explain the difference between M. Tech and M.Sc. in cybersecurity?",
"Are there any cybersecurity courses offered online by RACE?",
"What is the demand for cybersecurity professionals?",
"What is the duration of RACE’s cybersecurity programs?",
"When do you start new batches for Cybersecurity programs at RACE?",
"Is industry experience mandatory for joining AI programs of RACE?",
"Is the M. Tech in Cybersecurity program of RACE accredited?",
"How can I learn more about M. Tech in Cybersecurity program of RACE?",
"Want to learn more about M. Sc. in Cybersecurity program of RACE?",
"How can I learn more about PGD in Cybersecurity program of RACE?",
"Do you have certification programs in the cybersecurity domain of RACE?",
"How do you promote experiential learning in Cybersecurity programs of RACE?",
"Why do I learn M. Sc. in Cloud Architecture and Security at RACE?",
"Who can apply to M. Sc. in Cloud Architecture and Security program of RACE? ",
"What are the subjects taught under the M. Sc. in Cloud Architecture and Security of RACE?",
"What is the mode of program delivery of M. Sc. in Cloud Architecture and Security of RACE?",
"What is the demand for cloud architecture and security professionals?",
"What is the duration of Cloud Architecture and Security program of RACE?",
"When do you start the new batch for M.Sc. in Cloud Architecture and Security program?",
"Is industry experience mandatory for joining M. Sc. in Cloud Architecture and Security of RACE?",
"Is M. Sc. in Cloud Architecture and Security program of RACE accredited?",
"How do you promote experiential learning in M.Sc. in Cloud Architecture and Security of RACE?",
"Which are the short-term programs available at RACE?",
"When do short-term programs of RACE commence?",
"What are the criteria for certification programs of RACE?",
"How can a certification program of RACE beneficial for me?",
"Are there any program offered online by RACE?",
"Do I receive a recognized certificate for short-term programs of RACE? ",
"What is the duration of short-term programs of RACE?",
"How do I apply for the short-term programs of RACE?",
"Do you have specific trainers for certification programs of RACE?",
"Will I be considered an alumnus of RACE after completing a short-term program?"]





qnnoglobal = 0
questionasked = ""



#Here need to send the voice input result
def greeting(recognizer_result):
    print(f"Printing recognized result sent to predict: {recognizer_result}")
    #print(f"after recog:{recognizer_result}")
    
    #fm_output=fuzzymatcher(recognizer_result)
    
    
    #print(f"final output answer: {commands[fm_output[0]]}")
    
    P=model.transform([Cleaning_questions(recognizer_result)])
    predict1=clf1.predict(P)
    print (predict1)
    return predict1


@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')

#/getqn
@app.route("/getqn", methods=["GET", "POST"])
def qnnum():
    if request.method == "POST":
       # getting input with name = fname in HTML form geting question number
       first_name = request.form.get("fname")
       qnno = int(first_name)
       print(type(qnno))
       global question
       #question = fetchqnno(qnno)
       question = li[(qnno-1)]
       #assinging question number globally
       qnnoglobal = qnno
    return render_template('index.html', question = question)
    


@app.route("/getresponse", methods=["GET", "POST"])
def fetch_response():
    #Here req is the voice input question by the user
    req = request.get_json()
    print(f"Printing the type of file : {type(req)}")
    print(f"Getting the response from javascript SPEECH RECOGNITION : {req}")

    res = make_response(jsonify({"message" : "JSON recived"}), 200)

    recognizer_result = req
    global answer
    answer = greeting(recognizer_result)
    return render_template('index.html', question = question,answer = answer)

@app.route("/getans", methods=["GET", "POST"])
def getans():
    if request.method == "POST":
       # getting input with name = fname in HTML form geting question number
       getans = request.form.get("getans")
    return render_template('index.html', question = question,answer = answer)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

