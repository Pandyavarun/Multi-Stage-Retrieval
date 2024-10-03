from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_reranker(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def rerank_passages(query, passages, reranker_model, reranker_tokenizer):
    inputs = [query + " [SEP] " + passage for passage, _ in passages]
    tokenized_inputs = reranker_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = reranker_model(**tokenized_inputs)
        scores = outputs.logits.squeeze(-1).tolist()

    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [(passage[0], score) for passage, score in ranked_passages]
