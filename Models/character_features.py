import os
import re
import numpy as np
import pandas as pd



def make_features(data,data_size,max_chars_per_sentence,max_length_vocab):
  char_level_features = np.zeros((data_size,max_chars_per_sentence,max_length_vocab))
  
  for i in range(data_size):
    count = 0
    sent_features = np.zeros((max_chars_per_sentence,max_length_vocab))
    chars = list(data[i])
    
    for c in chars:
      if(count>=max_chars_per_sentence):
        break
      elif(c>='a' and c<='z'):
        feature = np.zeros(max_length_vocab)
        feature[ord(c)-97] = 1
        sent_features[count,:] = feature
        count+=1
    char_level_features[i,:,:] = sent_features
    
  return char_level_features
