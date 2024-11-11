import json
from sentence_transformers import SentenceTransformer
from config import config

def load_concepts():
    with open(config.CONCEPTS_FILE, 'r') as f:
        return json.load(f)

def get_concept_embeddings(filtered_concepts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    concept_embeddings = {concept: model.encode(concept, convert_to_tensor=True)
                          for concepts in filtered_concepts.values() for concept in concepts}
    return concept_embeddings

def get_concept_names(filtered_concepts):
    return [concept for concepts in filtered_concepts.values() for concept in concepts]

