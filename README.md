# Character-level-CNN
Implementation of Character level CNN

# Getting started
This repository consists of code for the implementation of Character Level CNN's proposed by Xiang Zhang,Junbo Zhao,Yann LeCun 
https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf. Character level CNN is useful when the word embeddings are not powerful enough to extract the relationship among words.In Character Level CNN a set of characters is considered as a dictionary each given an unique index.Each sentence is converted to a fixed size of 3D vector.Each character is one-hot encoded using the dictionary indices, if the character count is greater they are ignored. These are then trained using CNN.

# Implementation
I have used a dictionary size of 26 and considered 50 as the maximum character size for each sentence.Dataset considered is   rt-polarity 2.0 dataset which consists of 5331 positive reviews and 5331 negative reviews.
 
 # Usage
 1. Install dependencies using
 ```pip install -r requirements.txt```
 2. clone the repository using 
 ``` git clone https://github.com/avinashsai/Character-level-CNN.git```
 3. Change the folder using
 ```cd Models```
 4. Run the main file using
 ```python3 character_level_cnn.py```
 
 
