import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample dataset
data = {
    'title': [
        'The Matrix', 'John Wick', 'The Notebook',
        'Avengers: Endgame', 'Interstellar', 'Titanic',
        'Inception', 'Shutter Island', 'The Lion King'
    ],
    'description': [
        'A hacker discovers reality is a simulation',
        'An ex-hitman goes on a revenge spree',
        'A romantic story of a couple',
        'Superheroes unite to fight a universal villain',
        'A journey through space and time',
        'A tragic love story on the Titanic ship',
        'A thief enters dreams to plant ideas',
        'A detective investigates a psychiatric mystery',
        'A lion cub becomes king of the jungle'
    ]
}
df = pd.DataFrame(data)
# TF-IDF Vectorization of descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Function to recommend movies
def recommend(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
print("Recommended for 'Inception':")
print(recommend('Inception'))
