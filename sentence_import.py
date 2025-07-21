from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('/models/embedding/all-MiniLM-L6-v2')