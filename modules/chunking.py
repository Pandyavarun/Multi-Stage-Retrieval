def chunk_document(doc_text, max_chunk_size=200, overlap_size=50):
    words = doc_text.split()
    chunks = []
    
    for i in range(0, len(words), max_chunk_size - overlap_size):
        chunk = words[i:i + max_chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks

def create_passages(corpus):
    passages = []
    passage_id = 0
    for doc_id, doc in corpus.items():
        doc_chunks = chunk_document(doc["text"])
        for chunk in doc_chunks:
            passages.append({
                "doc_id": f"{doc_id}_{passage_id}",
                "text": chunk
            })
            passage_id += 1
    return passages
