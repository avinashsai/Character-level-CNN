import os
import re
import numpy as np
import pandas as pd
from preprocess import *

def load_files(filename):
  with open(filename,'r',encoding='latin1') as f:
    data = f.readlines()
  return data


def make_corpus(pos,neg,stopword):
  corpus = []
  labels = np.zeros(10662)
  
  for i in range(5331):
    corpus.append(preprocess(pos[i],stopword))
    
  for i in range(5331):
    corpus.append(preprocess(neg[i],stopword))
   
  labels[0:5331] = 1
  
  return corpus,labels

