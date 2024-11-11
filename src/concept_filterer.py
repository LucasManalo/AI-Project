import json
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path

# Define the path to the JSON file
project_root = Path(__file__).resolve().parents[1]  # Moves up two directories
concepts_path = project_root / "generated_concepts.json"

# Open the file
with open(concepts_path, 'r') as f:
    concepts_dict = json.load(f)

# Initialize sentence transformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Thresholds for filtering
MAX_CONCEPT_LENGTH = 30
SIMILARITY_THRESHOLD = 0.85  # Threshold for similarity filtering

def filter_concepts(concepts_dict, model, target_class_names):
    filtered_concepts = {}

    for class_name, concepts in concepts_dict.items():
        # Step 1: Remove long concepts
        concepts = [c for c in concepts if len(c) <= MAX_CONCEPT_LENGTH]

        # Generate embeddings for each concept
        concept_embeddings = model.encode(concepts, convert_to_tensor=True)

        # Step 2: Remove concepts too similar to the target class name
        class_embedding = model.encode(class_name, convert_to_tensor=True)
        keep_indices = [i for i, emb in enumerate(concept_embeddings)
                        if util.cos_sim(emb, class_embedding) < SIMILARITY_THRESHOLD]
        concepts = [concepts[i] for i in keep_indices]
        concept_embeddings = concept_embeddings[keep_indices]

        # Step 3: Remove duplicate or very similar concepts
        unique_concepts = []
        unique_embeddings = []
        for i, concept in enumerate(concepts):
            if all(util.cos_sim(concept_embeddings[i], ue) < SIMILARITY_THRESHOLD for ue in unique_embeddings):
                unique_concepts.append(concept)
                unique_embeddings.append(concept_embeddings[i])

        # Save the final filtered concepts for this class
        filtered_concepts[class_name] = unique_concepts

    return filtered_concepts

# List of target classes
target_classes = list(concepts_dict.keys())
filtered_concepts = filter_concepts(concepts_dict, model, target_classes)

# Print or save the filtered concepts
with open('filtered_concepts.json', 'w') as f:
    json.dump(filtered_concepts, f, indent=4)

print("Filtered concepts saved to 'filtered_concepts.json'")