import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

root = '/raid/mlcysec19/student_directories/team9/vec/'
files = os.listdir(root)

contents = [open(root + f) for f in files]
vectorizer = CountVectorizer(input='file',
                        token_pattern='[0-9a-fA-F]+', # tokens aren't detected unless
                                                      # we use this regex
                        max_features=100,
                        ngram_range=(2,2))

# By changing `max_features`, which determines dimensionality of representation
# and `ngram_range`, which deteremines upto what level of ngrams we consider
# we can create many different vectorizer schemes.

# We tried following
# 1. Unigram 25 dimension (max_features=25, ngram_range=(1,1))
# 2. Unigram 50 dimension (max_features=50, ngram_range=(1,1))
# 3. Unigram 100 dimension (max_features=100, ngram_range=(1,1))
# 4. Unigram 200 dimension (max_features=200, ngram_range=(1,1))
# 5. Unigram+Bigram 50 dimension (max_features=50, ngram_range=(1,2))
# 6. Bigram 100 dimension (max_features=100, ngram_range=(2,2))

X = vectorizer.fit_transform(contents)

# Here we save the vectorizer file, which we then load in PyTorch transformers.
vfh = open('bigrams_100.v','wb')
pickle.dump(vectorizer, vfh)
vfh.close()

# According to MIST Format documentation only the first few arguments are importants
# as we go more to the right, arguments become much less informative.
# After doing a static analysis of token frequencies we tried following 6 different
# representations:
# 1. Unigram with 25 dimensions
# 2. Unigram with 50 dimensions
# 3. Unigram with 100 dimensions
# 4. Unigram with 200 dimensions
# 5. Unigram+Bigram v 50 dimensions
# 6. Bigram with 100 dimensions
