import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

class GPT2Embedder:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.word_embeddings = self.model.transformer.wte.weight  # Word Token Embeddings 
        self.position_embeddings = self.model.transformer.wpe.weight  # Word Position Embeddings
    def embed_corpus(self, list_of_strings):
        self.text_index = self.tokenizer.encode(list_of_strings, add_prefix_space=True)
        self.vector = self.model.transformer.wte.weight[text_index, :]
        return vector
    
class BertEmbedder:
    def __init__(self, version):
        self.model = SentenceTransformer(version)
    def embed_corpus(self, sentences):
        return self.model.encode(sentences)