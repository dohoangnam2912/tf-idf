import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextSimilarity:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df['text'].tolist()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
    
    def get_similar_words(self, input_text):
        input_tfidf = self.vectorizer.transform([input_text])
        similarities = cosine_similarity(input_tfidf, self.tfidf_matrix)
        most_similar_indices = similarities[0].argsort()[-10:][::-1]

        most_similar_texts = [self.texts[i] for i in most_similar_indices]
        scores = [similarities[0][i] for i in most_similar_indices]
        return most_similar_texts, scores

    def print_similar_texts(self, input_text):
        similar_texts, scores = self.get_similar_words(input_text)
        print("Input Text:", input_text)
        print("\nSimilar Texts and Scores:")
        for text, score in zip(similar_texts, scores):
            print(f"Score: {score:.4f} - Text: {text}")

# Usage
csv_file = 'testing_vn.csv'
text_similarity = TextSimilarity(csv_file)
input_text = "Tôi yêu mèo"
text_similarity.print_similar_texts(input_text)


