from sentence_transformers import SentenceTransformer, util

def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def embed_passages(passages, model):
    return model.encode(passages, convert_to_tensor=True)

def retrieve_top_k(query, passages, embeddings, model, k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_k_indices = scores.argsort(descending=True)[:k]
    return [(passages[idx], scores[idx].item()) for idx in top_k_indices]
