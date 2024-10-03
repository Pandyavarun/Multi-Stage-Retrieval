# app.py

from flask import Flask, render_template, request, jsonify
from modules.embedding import load_embedding_model, embed_passages, retrieve_top_k
from modules.reranking import load_reranker, rerank_passages
from modules.dataset_loader import load_dataset
from modules.chunking import create_passages

app = Flask(__name__)

# Load models
embedding_model = load_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
reranker_model, reranker_tokenizer = load_reranker('cross-encoder/ms-marco-MiniLM-L-12-v2')

# Load dataset and preprocess passages
corpus, queries, qrels = load_dataset('natural-questions')
passages = create_passages(corpus)
passage_texts = [p['text'] for p in passages]
passage_embeddings = embed_passages(passage_texts, embedding_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    top_k_passages = retrieve_top_k(query, passage_texts, passage_embeddings, embedding_model, k=10)
    reranked_passages = rerank_passages(query, top_k_passages, reranker_model, reranker_tokenizer)
    
    return jsonify({"results": reranked_passages})

if __name__ == '__main__':
    app.run(debug=True)
