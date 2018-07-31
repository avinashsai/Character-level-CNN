import os
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.utils import shuffle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from character_features import *
from loadfiles import *
from model import *
from preprocess import *

nltk.download('punkt')
nltk.download('stopwords')


def main():
	pos = load_files('../Datasets/rt-polarity/rt-polarity.pos')
	neg = load_files('../Datasets/rt-polarity/rt-polarity.neg')

	stopword = stopwords.words('english')
	corpus,labels = make_corpus(pos,neg,stopword)
	print(corpus[0:2])

	train_d,test_d,train_l,test_l = train_test_split(corpus,labels,test_size=0.3,random_state=42)
	train_d,train_l = shuffle(train_d,train_l)
	test_d,test_l = shuffle(test_d,test_l)
	train_len = len(train_l)
	test_len = len(test_l)
	print(train_len)
	print(test_len)

	max_chars_per_sentence = 50 
	max_length_vocab = 26


	print("Getting Features:")
	train_vecs_features = make_features(train_d,train_len,max_chars_per_sentence,max_length_vocab)
	test_vecs_features = make_features(test_d,test_len,max_chars_per_sentence,max_length_vocab)

	print("Running model:")
	pred = run_model(train_vecs_features,test_vecs_features,train_l)
	y_pred = np.zeros(test_len)

	for i in range(test_len):
		if(pred[i]>0.5):
			y_pred[i] = 1

	print("Accuracy :",accuracy_score(y_pred,test_l))
	print("F1 Score :",f1_score(y_pred,test_l))

if __name__ == '__main__':
	main()