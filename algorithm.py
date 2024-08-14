import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

csv_file = 'testing_vn.csv' 
df = pd.read_csv(csv_file)

texts = df['text'].tolist()
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(texts)

dense_tfidf_matrix = tfidf_matrix.toarray()

def get_similar_words(input_text):
    input_tfidf = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    most_similar_indices = similarities[0].argsort()[-10:][::-1]

    most_similar_texts = [texts[i] for i in most_similar_indices]
    scores = [similarities[0][i] for i in most_similar_indices]
    return most_similar_texts, scores

input_text = "Tôi yêu mè"
similar_texts, scores = get_similar_words(input_text)


print("Input Text:", input_text)
print("\nSimilar Texts and Scores:")

for text, score in zip(similar_texts, scores):
    print(f"Score: {score:.4f} - Text: {text}")

