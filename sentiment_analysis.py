#loading python librairies
import pandas as pd
from pandas.core.common import flatten
import numpy as np
import seaborn as sns
import unicodedata
from tqdm.notebook import tqdm_notebook

from matplotlib import pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding,RNN,Dropout,GlobalAveragePooling1D,Dense,GRUCell
from tensorflow.keras import callbacks
from tensorflow.keras import models

import mlflow
import mlflow.keras
import mlflow.tensorflow

#loading data set
#The data set is available here : https://www.kaggle.com/kazanova/sentiment140
df=pd.read_csv("train.csv",encoding="latin-1",header = None)

df.columns=['sentiment', 'id', 'date', 'query', 'user', 'tweet']
df = df.drop(columns=['id', 'date', 'query', 'user'], axis=1)
df.head()
#replacing the label 4 with 1 
df["sentiment"]= df["sentiment"].replace(4,1)
df.head()

#counting des number of labels and their frequencies in data set
df["sentiment"].value_counts()

#counting the pourcentage of each label
df["sentiment"].value_counts(normalize=True)
#shwoing the proportion of labels in data set
plt.style.use('ggplot')
sns.countplot(x="sentiment",data=df)

#Cleaning the data set
stop_words = stopwords.words('english')

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)
    #removing the linkes
    w = re.sub(r"https?://[a-zA-Z0-9./]+", " ", w)
    #removing mails
    w = re.sub(r"[a-zA-Z0-9.-]+@[a-zA-Z.]+", " ", w)
    #removing htags
    w = re.sub(r"#[a-zA-Z0-9]+", "_", w)
    w = re.sub(r"@[a-zA-Z0-9]+", " ", w)
    # remove stopword
    words = word_tokenize(w.strip())
    words = [mot for mot in words if mot not in stop_words]
    return words

tqdm_notebook.pandas()
df["tweet"]= df["tweet"].progress_apply(lambda x :preprocess_sentence(x))
df.head()
#word cloud for the tweets with sentiment=0
wc=WordCloud(background_color="black",max_words=100,stopwords=stop_words,max_font_size=50)
text_0=""

for comment in df["tweet"][df["sentiment"]==0]:
    comment=' '.join(comment)
    text_0+=comment

plt.figure(figsize= (10,6)) 
wc.generate(text_0)           
plt.imshow(wc) 
plt.show()
#word cloud for the tweets with sentiment=1
text_1=""
for comment in df["tweet"][df["sentiment"]==1]:
    comment=' '.join(comment)
    text_1+=comment


plt.figure(figsize= (10,6)) 
wc.generate(text_1)           
plt.imshow(wc) 
plt.show()

#plot distribution for the lenght of tweets
sns.displot(list(map(len,df["tweet"])))

#plot distribution for the tweet lenghts with sentiment=0
sns.displot(list(map(len,df["tweet"][df["sentiment"]==0])))

#plot distribution for the tweet lenghts with sentiment=1
sns.displot(list(map(len,df["tweet"][df["sentiment"]==1])))

#dividing the data set into test and train
X_train, X_test, y_train, y_test = train_test_split(df["tweet"], df["sentiment"], test_size=0.2, random_state=1234)

#vectorizing tweets
words_unique=list(set(flatten(df["tweet"].values)))
print(len(words_unique))

tokenizer=Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)

X_train=pad_sequences(X_train,maxlen=20,padding="post",truncating="post")
X_test=pad_sequences(X_test,maxlen=20,padding="post",truncating="post")

#definig our RNN model
words_unique=list(set(flatten(df["tweet"].values)))
num_words=len(words_unique)
max_len = 20
print(num_words)

model=Sequential()
model.add(Embedding(input_dim=num_words,output_dim=max_len,input_length=max_len))
model.add(RNN(GRUCell(128),return_sequences=True))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling1D())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2,activation="sigmoid"))
model.summary()

#compiling the model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#definig the callbacks
lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'loss',
                                             patience = 2,
                                             factor = 0.2,
                                             verbose = 2,
                                             mode = 'min',
                                             min_lr=0)
early_stopping = callbacks.EarlyStopping(monitor = "loss",
                                             patience = 3,
                                             mode = 'min',
                                             verbose = 2,
                                             restore_best_weights= True)
checkpoint = callbacks.ModelCheckpoint(filepath="C:/Users/Sevil/Desktop/datascientest/data_sets/corpus_sentiment_analysis/check",
                                          monitor = 'loss',
                                          save_best_only = True,
                                          save_weights_only = False,
                                          mode = 'min',
                                          save_freq = 'epoch')

mlflow.set_experiment("sentiment_analysis")
epochs=10
with mlflow.start_run() as run:
    
    history = model.fit(X_train,y_train,epochs = epochs,validation_data=(X_test,y_test),callbacks=[lr_plateau,early_stopping, checkpoint],use_multiprocessing= True,batch_size=100)
    
    mlflow.tensorflow.autolog(every_n_iter=1)
    
    probs = model.predict(X_test)
    y_pred = np.argmax(probs, axis=1)
    fscore=f1_score(y_test,y_pred)   
    mlflow.log_metric("f1-score",fscore)
    mlflow.log_param("epochs",epochs)
    model_name = "sentiment_analysis"
    artifact_path="artifacts"
    mlflow.keras.log_model(keras_model=model, artifact_path=artifact_path)
    mlflow.keras.save_model(keras_model=model, path=model_name)
    mlflow.log_artifact(local_path=model_name)
    runID=run.info.run_uuid
    mlflow.register_model("runs:/"+runID+"/"+artifact_path,"sentiment_analysis")

#model evaluation
model.evaluate(X_test,y_test.values,batch_size=128)

#confusion matrix
probs = model.predict(X_test)
y_pred = np.argmax(probs, axis=1)

conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print(conf_matrix)
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0,1])

disp.plot()
plt.show()

#plot loss and accuracy for test and train

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
plt.show()
