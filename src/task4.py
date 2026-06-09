import numpy as np
import pandas as pd
import random


# load preprocessed data from Task 2/3
df = pd.read_json('processed.json', orient='records', lines=True)

# extract tokens (split the processed strings back into lists)
sentences = [str(text).split() for text in df['Processed_Text']]

# count word occurence to remove rare words to remove noise
word_counts = {}
for sent in sentences:
    for word in sent:
        word_counts[word] = word_counts.get(word, 0) + 1

# keep only words that appear at least 5 times
vocab = [w for w, count in word_counts.items() if count >= 5]
w2i = {w: i for i, w in enumerate(vocab)}  # Word to Index
i2w = {i: w for i, w in enumerate(vocab)}  # Index to Word
V = len(vocab)

print(f"Original vocabulary size: {len(word_counts)}")
print(f"Filtered vocabulary size (freq >= 5): {V}")


# initialize an empty V x V matrix with zeros
C = np.zeros((V, V), dtype=np.float32)

for sent in sentences:
    for i in range(len(sent) - 1):
        w1, w2 = sent[i], sent[i+1]
        
        # only count if both words are used enough 
        if w1 in w2i and w2 in w2i:
            id1, id2 = w2i[w1], w2i[w2]
            # window size 1, we count both directions to make the matrix symmetric
            C[id1, id2] += 1
            C[id2, id1] += 1



# PMI Formula: log2( P(x,y) / (P(x) * P(y)) )
total_cooc = C.sum()

# P(x,y) = Probability of x and y appearing together
P_xy = C / total_cooc

# P(x) and P(y) = Probability of x and y appearing individually
# np.sum(axis=1) sums up the rows, keeping it a 1D array
P_x = C.sum(axis=1) / total_cooc
P_y = C.sum(axis=0) / total_cooc

# np.outer calculates P(x) * P(y) for all combination 
P_x_P_y = np.outer(P_x, P_y)

# calculate PMI, ignoring div and log of 0 
with np.errstate(divide='ignore', invalid='ignore'):
    PMI = np.log2(P_xy / P_x_P_y)
    
    # replace NaN and all negative values and -inf with 0
    PMI[np.isnan(PMI)] = 0.0
    PMI[PMI < 0] = 0.0


# to calculate cosine similarity quickly, we normalize the PMI matrix vectors to length 1
# np.linalg.norm calculates the length of each row vector
norms = np.linalg.norm(PMI, axis=1, keepdims=True)
norms[norms == 0] = 1e-9  # Prevent division by zero

PMI_norm = PMI / norms

# pick 10 random words
random.seed(20)  # for reproducibility
sample_words = random.sample(vocab, 10)

print(f"\n{'='*50}\nTop 5 Similar Words based on PMI\n{'='*50}")

for word in sample_words:
    word_id = w2i[word]
    word_vec = PMI_norm[word_id]
    
    # cosine Similarity between our word and all other words 
    # because vectors are normalized, dot product = cosine similarity
    sims = np.dot(PMI_norm, word_vec)
    
    # .argsort() sorts from lowest to highest we take the last 6 (highest)
    # [::-1] reverses it so the highest is first
    top_indices = sims.argsort()[-6:][::-1]
    
    # exclude the word itself 
    top_indices = [idx for idx in top_indices if idx != word_id][:5]
    
    print(f"\nTarget Word: [{word.upper()}]")
    for idx in top_indices:
        sim_word = i2w[idx]
        score = sims[idx]
        print(f"  -> {sim_word:<15} (Score: {score:.3f})")
