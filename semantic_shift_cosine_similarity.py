# ====================================================
# Semantic Shift Analysis of Economic Terms
# Python implementation for cosine similarity between
# word vectors across 2005, 2009, and 2011
# ====================================================

# Import required packages
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



# Define word vectors for each year and term
# Each vector contains weighted values (MI)
# representing the collocational profile of the word
vectors = {
    "Economy": {
        2005: np.array([0.693, 0.693, 0.693, 0.693, 1.792, 2.197, 0.693, 0.693, 0.693, 0.693,
    0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 1.792, 0.693,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0]),
        2009: np.array([0, 0, 0, 0, 1.386, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1.099, 0.693, 0.693, 0.693, 0.693, 1.099, 0.693, 0.693, 0.693, 0.693,
    0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 0.693, 1.609, 0,
    0, 0, 0, 0, 0, 0, 0, 0]),
        2011: np.array([0, 0, 0, 0, 0.693, 2.197, 0, 0.693, 0, 0,
    0, 0, 0, 0, 0.693, 0, 0, 0, 2.303, 0,
    0.693, 0.693, 0, 0, 0, 0.693, 0, 0, 0, 0,
    0, 0.693, 0, 0.693, 0, 0, 0, 0.693, 0, 0.693,
    0.693, 0.693, 0.693, 1.099, 0.693, 0.693, 0.693, 0.693])
    },
    "Job": {
        2005: np.array([0.693,0.693,1.099,1.386,1.099,0.693,
    0.693,0.693,0.693,1.099,0.693,0.693,2.303,
    1.099,1.099,0.693,0.693,0.693,1.386,2.079,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        2009: np.array([0.693,0,0.693,1.386,1.099,0,0,0.693,0.693,
    1.792,0,0,2.079,1.386,0,
    1.609,0,0,0,0,0.693,1.099,0.693,0.693,0.693,0.693,0.693,2.079,0.693,
    1.099,0,0,0,0,0,0,0,0]),
        2011: np.array([0.693,0,0,0.693,1.099,0,0,
    0.693,0.693,1.609,0,0,1.946,1.609,0,0.693,0,0,0,0,
    0.693,0.693,0,0,0,0,0,0,
    0.693,0,0.693,1.099,0.693,0.693,0.693,1.792,1.386,0.693])
    },
    "Crisis": {
        2005: np.array([0.693,0.693,0.693,0.693,0.693,0.693,1.099,0.693,0.693,0.693,
    1.099,0.693,0.693,0.693,0.693,0.693,1.099,0.693,0.693,0.693,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        2009: np.array([0,0.693,0.693,0.693,0,0,
    2.996,0,1.099,0,0,1.099,0,2.708,0,1.946,0,0,0,0,
    0.693,0.693,0.693,0.693,0.693,0.693,0.693,0.693,2.079,1.609,0.693,0.693,
    0,0,0,0,0,0,0,0,0,0,0,0,0]),
        2011: np.array([0.693,0,0,0.693,0,0,2.708,0,0,0,0,0,0,
    1.792,0,1.609,0,0,0,0,0,0,0,0.693,0.693,0,0,0,0,0,0,0,
    0.693,0.693,0.693,1.386,0.693,0.693,0.693,0.693,0.693,0.693,0.693,0.693,0.693])
    }
}



# Reshape vectors to 2D arrays
# sklearn's cosine_similarity expects 2D arrays
for term in vectors:
    for year in vectors[term]:
        vectors[term][year] = vectors[term][year].reshape(1, -1)



# Compute cosine similarity and store results in a dictionary
# Cosine similarity = measure of how similar two vectors are
# 1.0 = identical vectors, 0 = completely different
similarity = {}

for term, year_vectors in vectors.items():
    similarity[term] = {}
    # 2005 vs 2009
    similarity[term]["2005_2009"] = cosine_similarity(year_vectors[2005], year_vectors[2009])[0][0]
    # 2009 vs 2011
    similarity[term]["2009_2011"] = cosine_similarity(year_vectors[2009], year_vectors[2011])[0][0]
    # 2005 vs 2011
    similarity[term]["2005_2011"] = cosine_similarity(year_vectors[2005], year_vectors[2011])[0][0]



#Print the results
for term, sims in similarity.items():
    print(f"\nCosine similarities for '{term}':")
    for period, value in sims.items():
        print(f"  {period.replace('_', ' vs ')}: {value:.4f}")
