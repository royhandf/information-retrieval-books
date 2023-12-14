from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv('Books.csv')

columns_for_search = ['Book-Title', 'Book-Author']

combined_columns = data[columns_for_search].fillna('').apply(lambda x: ' '.join(x), axis=1)

vectorizer = TfidfVectorizer()
book_vectors = vectorizer.fit_transform(combined_columns)

def search_book(query, top_n=5):
    query_vector = vectorizer.transform([query])

    cosine_similarities = cosine_similarity(query_vector, book_vectors)

    top_indices = cosine_similarities[0].argsort()[-top_n:][::-1]
    
    results = []
    
    for index in top_indices:
        results.append({
            "isbn": data['ISBN'][index],
            "title": data['Book-Title'][index],
            "author": data['Book-Author'][index],
            "publisher": data['Publisher'][index],
            "year": data['Year-Of-Publication'][index],
            "image_url": data['Image-URL-L'][index],
        })  
    return results

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]
        results = search_book(query)
        return render_template("results.html", query=query, results=results)
    else:
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)