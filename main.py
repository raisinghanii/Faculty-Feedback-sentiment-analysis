#import libraries....        
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
import re
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, render_template
import os
from matplotlib import pyplot as plt
import h5py


#preprocessing....
revs = pickle.load(open('pre2','rb'))
max_fatures = 70000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(revs)
X1 = tokenizer.texts_to_sequences(revs)
X1 = pad_sequences(X1)

#load model....
model = load_model('model2.h5')
model.summary()


#analyze....
def analyze(str,model=model):
    rev = re.sub('[^a-zA-Z]', ' ',str)
    rev = rev.lower()
    rev = rev.split()
    ps = PorterStemmer()
    rev = [ps.stem(word) for word in rev]
    qq = tokenizer.texts_to_sequences(rev)
    lst = [item for elem in qq for item in elem]
    pp = []
    pp.append(lst)
    padded = pad_sequences(pp,maxlen=1360)
    pred = model.predict(padded)
    return pred[0][0], pred[0][1]
    #print("Negative :",pred[0][0],"\tPositive :",pred[0][1])
    
    


#the frontend part.....yet to develop.....




app = Flask(__name__)


def createfdir(dirname):
    parent_dir = "C:/Users/Yogesh/Desktop/Feedback/Faculty"
    path = os.path.join(parent_dir, dirname)
    if not os.path.exists(path):
        os.makedirs(path)


def createcdir(dirname):
    parent_dir = "C:/Users/Yogesh/Desktop/Feedback/Course"
    path = os.path.join(parent_dir, dirname)
    if not os.path.exists(path):
        os.makedirs(path)


def savef(fname, pos, neg):
    pdir = 'C:/Users/Yogesh/Desktop/Feedback/Faculty/{}/positive.txt'.format(fname)
    ndir = 'C:/Users/Yogesh/Desktop/Feedback/Faculty/{}/negative.txt'.format(fname)

    posfile = open(pdir, 'a')
    posfile.write(str(pos) + '\n')
    posfile.close()

    negfile = open(ndir, 'a')
    negfile.write(str(neg) + '\n')
    negfile.close()

def savec(fname, pos, neg):
    pdir = 'C:/Users/Yogesh/Desktop/Feedback/Course/{}/positive.txt'.format(fname)
    ndir = 'C:/Users/Yogesh/Desktop/Feedback/Course/{}/negative.txt'.format(fname)

    posfile = open(pdir, 'a')
    posfile.write(str(pos) + '\n')
    posfile.close()

    negfile = open(ndir, 'a')
    negfile.write(str(neg) + '\n')
    negfile.close()


def readfpie(fname):
    pdir = 'C:/Users/Yogesh/Desktop/Feedback/Faculty/{}/positive.txt'.format(fname)
    ndir = 'C:/Users/Yogesh/Desktop/Feedback/Faculty/{}/negative.txt'.format(fname)
    idir = 'C:/Users/Yogesh/Desktop/Feedback/Faculty/{}/piechart.png'.format(fname)

    rp = open(pdir, 'r')
    pos = rp.readlines()
    sump = 0
    for line in pos:
        sump += float(line)
    print(sump)

    rn = open(ndir, 'r')
    neg = rn.readlines()
    sumn = 0
    for line in neg:
        sumn += float(line)
    print(sumn)

    revs = ['Positive', 'Negative']
    colours = ['green', 'red']
    data = [sump, sumn]

    fig = plt.figure(figsize=(10, 7))
    plt.pie(data, labels=revs, colors=colours)

    plt.savefig(idir)

def readcpie(fname):
    pdir = 'C:/Users/Yogesh/Desktop/Feedback/Course/{}/positive.txt'.format(fname)
    ndir = 'C:/Users/Yogesh/Desktop/Feedback/Course/{}/negative.txt'.format(fname)
    idir = 'C:/Users/Yogesh/Desktop/Feedback/Course/{}/piechart.png'.format(fname)

    rp = open(pdir, 'r')
    pos = rp.readlines()
    sump = 0
    for line in pos:
        sump += float(line)
    print(sump)

    rn = open(ndir, 'r')
    neg = rn.readlines()
    sumn = 0
    for line in neg:
        sumn += float(line)
    print(sumn)

    revs = ['Positive', 'Negative']
    colours = ['green', 'red']
    data = [sump, sumn]

    fig = plt.figure(figsize=(10, 7))
    plt.pie(data, labels=revs, colors=colours)

    plt.savefig(idir)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    fname = request.form['fname']
    frev = request.form['frev']
    cname = request.form['cname']
    crev = request.form['crev']

    print(fname, frev, cname, crev)

    createfdir(fname)
    negf,posf = analyze(frev,model)
    savef(fname, posf, negf)
    readfpie(fname)

    createcdir(cname)
    negc,posc = analyze(crev,model)
    savec(cname, posc, negc)
    readcpie(cname)



    return "Your Feedback has been recorded..."


if __name__ == '__main__':
    app.run(host="localhost", port=int("777"),debug=True)
