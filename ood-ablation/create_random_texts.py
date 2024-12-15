'''
File to create a small dataset of random words from the Gutenberg corpus 
to test against the pre-trained and our Vec2Text models. 

Resource Used:
https://www.nltk.org/book/ch02.html

Author: Emily Weiss
'''

import nltk
nltk.download('gutenberg')

from collections import Counter
import random

# get names of all texts in Gutenberg 
texts = nltk.corpus.gutenberg.fileids()

# make a list of unique, lowercase, alphabetic words from the Gutenberg corpus 
c = Counter()

for text in texts:
    words = nltk.corpus.gutenberg.words(text)
    for word in words:
        if word.isalpha():
            c[word.lower()] += 1

word_list = list(c)

# magic numbers 
num_rows = 15000
max_tokens = 32
min_tokens = 1

# make list of new, random texts 
new_texts = []
for i in range(num_rows):
    num_words = random.randrange(min_tokens, max_tokens)
    new_text = ""

    for j in range(num_words):
        new_text = new_text + random.choice(word_list) + " "

    new_texts.append(new_text)

# save random texts to file 
with open("random_texts_256.txt", "w") as f:
    for text in new_texts:
        f.write(text)
        f.write("\n")
