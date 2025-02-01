import json                         # Used to load food data from a JSON file.
import chromadb                     # A vector database for storing and retrieving embeddings.
from transformers import pipeline   # Library that allows us to use pre-trained AI models (like ChatGPT).
from sentence_transformers import SentenceTransformer  # A model for generating text embeddings.


# Load food data from JSON file
with open("foodDataSet.json", "r") as file:  #Reads a file called FoodDataSet.json (which contains food items).
    food_items = json.load(file)             #Loads the data into a Python list called food_items.

# Initialize ChromaDB client for Storing Food Data
client = chromadb.Client() #Initializes a ChromaDB client (database).
collection_name = "food_collection"  #Name of the collection where we will store food embeddings.

# Check if collection already exists, create new collection if not
if collection_name not in client.list_collections():
    collection = client.create_collection(collection_name)
else:
    collection = client.get_collection(collection_name)

# Define the sample query
query = "I want to eat curry for dinner"

#Load a Pre-trained AI Model for Food Classification
# Initialize Hugging Face zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") #Uses an AI model (facebook/bart-large-mnli) to classify text into different food categories.

# Classify the User Query into Dietary Categories
labels = ["vegan", "vegetarian", "non-vegan", "non-vegetarian", "pescatarian"] #The possible dietary categories.
classification_result = classifier(query, candidate_labels=labels)  # The AI guesses which category the query fits into.

# Print the classification result
print("Classification Result:")
for label, score in zip(classification_result['labels'], classification_result['scores']):
    print(f"  - {label.capitalize()}: {score:.2f}")
print("\n" + "="*50 + "\n")

# Use SentenceTransformer to generate embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #lightweight AI model specialized in finding text similarities.

# Prepare documents for food items (combining food name and description)
documents = [f"{item['food_name']}: {item['food_description']}" for item in food_items]
food_embeddings = embedder.encode(documents)

# Check if a Food Item Already Exists in ChromaDB
def check_existing_id(collection, food_id):
    # Fetch current documents in the collection
    collection_data = collection.get()
    
    # Check if 'ids' exists and search for the food_id in it
    if 'ids' in collection_data:
        existing_ids = collection_data['ids']
        return food_id in existing_ids
    else:
        # If 'ids' key is not found, handle it accordingly
        print("No IDs found in the collection.")
        return False

# Add food items to the collection, skipping duplicates
for item in food_items:
    food_id = str(item['food_id'])

    if not check_existing_id(collection, food_id):
        # Only add if the ID is unique
        collection.add(
            ids=[food_id],
            documents=[f"{item['food_name']}: {item['food_description']}"],
            embeddings=[food_embeddings[food_items.index(item)]]
        )
    else:
        print(f"Skipped {item['food_name']} because it's already in the collection.")


# Function to perform similarity search ( Search for Similar Food Items Based on Query)
def perform_similarity_search(query, collection):
    query_embedding = embedder.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    # Process results
    food_ids = results['ids'][0]
    distances = results['distances'][0]
    top_food_items = []

    for index, (food_id, score) in enumerate(zip(food_ids, distances), start=1):
        # Find matching food item by ID
        food_item = next((item for item in food_items if str(item['food_id']) == food_id), None)
        if food_item:
            top_food_items.append({
                "food_name": food_item['food_name'],
                "score": score,
                "food_description": food_item['food_description']
            })

    return top_food_items

# Perform similarity search for the query
top_food_items = perform_similarity_search(query, collection)

# Display the top 5 recommended food items
print("\nTop 5 Recommended Food Items Based on Your Query:")
for index, item in enumerate(top_food_items, start=1):
    print(f"{index}. {item['food_name']}")
    print(f"   Score: {item['score']:.4f}")
    print(f"   Description: {item['food_description']}")
    print("-" * 50)

'''
Summary
What This Code Does
Loads food data from a JSON file.
Initializes ChromaDB (a vector database).
Classifies the user query into a food category using Hugging Face's BART model.
Generates embeddings for food items using all-MiniLM-L6-v2.
Stores embeddings in ChromaDB (avoids duplicates).
Performs a similarity search to find the most relevant food items.
Prints the top 5 recommendations.
'''