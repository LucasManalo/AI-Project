import openai
import json
import os

# The API key will automatically be picked up from the environment variable OPENAI_API_KEY
openai.api_key = open(os.path.join(os.path.expanduser("~"), ".openai_api_key"), "r").read()[:-1]

# CIFAR-10 classes
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

def generate_concepts_for_class(class_name):
    prompt = f"List the most important features for recognizing something as a '{class_name}'. These should be visually distinguishable features. Only provide the feature itself and do not provide an explanation of why the feature fits with with the object '{class_name}'. Generate a list of 20 such features for the class '{class_name}'."
    # Call OpenAI's API using the chat completion method
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Or use "gpt-3.5-turbo" if you'd like
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=750,  # Adjust based on desired length of response
        temperature=0.7  # Temperature controls creativity; 0.7 is moderate
    )

    # Extract the concepts from the response
    concepts = response['choices'][0]['message']['content'].strip().split("\n")
    return [concept.strip() for concept in concepts if concept.strip()]

def generate_concepts_for_all_classes(class_names):
    concepts_dict = {}
    
    for class_name in class_names:
        print(f"Generating concepts for: {class_name}")
        concepts = generate_concepts_for_class(class_name)
        concepts_dict[class_name] = concepts
        print(f"Concepts for {class_name}: {concepts}\n")
    
    return concepts_dict

if __name__ == "__main__":
    # Generate concepts for CIFAR-10 classes
    concepts = generate_concepts_for_all_classes(cifar10_classes)
    
    # Save the generated concepts to a JSON file for later use
    with open('generated_concepts.json', 'w') as f:
        json.dump(concepts, f, indent=4)
    
    print("Concepts generated and saved to 'generated_concepts.json'")
