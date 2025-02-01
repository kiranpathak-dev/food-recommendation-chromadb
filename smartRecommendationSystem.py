'''
This code implements a recipe recommendation system that extracts ingredients from a PDF recipe, 
generates embeddings for them, and then finds similar recipes stored in ChromaDB 
using cosine similarity.
'''
import json
import chromadb
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import PyPDF2          #Extracts text from a PDF file (recipe document).

# Extract Text from a PDF File
'''
How it Works
Opens the PDF file in binary mode (rb).
Reads all pages and extracts text using page.extract_text().
Removes extra spaces and newlines.
Returns the extracted text.
If an error occurs (e.g., the file doesn't exist), it prints an error message and returns an empty string.
'''

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        with open(file_path, "rb") as file:  # Open the PDF file in binary mode
            reader = PyPDF2.PdfReader(file)   # Create a PDF reader object
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            text = " ".join(text.split())  # Remove extra spaces and newlines
        return text
    except Exception as e:
        print("Error extracting text from PDF:", e)
        return ""

#Generate Embeddings Using a Pre-trained Model
'''How it Works
Uses the Sentence Transformer model to convert text into embeddings.
Converts the generated embeddings to tensor format and then to a Python list.
If an error occurs, it prints an error message and returns None.
'''
def generate_embeddings(text, model):
    """Generates embeddings using a pre-trained SentenceTransformer model."""
    try:
        embedding = model.encode(text, convert_to_tensor=True).tolist()
        return embedding
    except Exception as e:
        print("Error generating embeddings:", e)
        return None

#Extract Ingredients from Text
'''How it Works
Uses Regular Expressions (re.search) to find ingredients mentioned in the text.
The pattern searches for “Ingredients” followed by any text until “For”.
If the pattern is found:
Extracts the ingredient list.
Splits it into words and converts them to lowercase.
Returns the list of ingredients.
If no match is found, it returns an empty list.
'''
def extract_ingredients(text):
    """Extracts ingredients from the given text."""
    import re
    ingredients_pattern = re.search(r"Ingredients([\s\S]+?)For ", text, re.IGNORECASE)
    if ingredients_pattern:
        ingredients = ingredients_pattern.group(1).split()
        return [ingredient.strip().lower() for ingredient in ingredients if ingredient.strip()]
    return []

# Store Food Embeddings in ChromaDB
'''How it Works
Extracts food names from the dataset.
Iterates through each food item:
Joins the ingredient list into a single text string.
Generates embeddings using the generate_embeddings function.
If successful, stores the embeddings.
Creates unique IDs for each food item.
Stores IDs, food names, and embeddings in ChromaDB.
'''
def store_embeddings_in_chromadb(food_items, collection, model):
    """Stores food embeddings in ChromaDB."""
    food_embeddings = []
    food_texts = [item['food_name'] for item in food_items]
    
    for item in food_items:
        ingredients_text = " ".join(item['food_ingredients']).lower()
        embedding = generate_embeddings(ingredients_text, model)
        if embedding:
            food_embeddings.append(embedding)
    
    ids = [str(index) for index in range(len(food_items))]
    collection.add(ids=ids, documents=food_texts, embeddings=food_embeddings)
    print("Stored embeddings in ChromaDB.")

'''
How it Works
Loads the food dataset (FoodDataSet.json).
Initializes ChromaDB and creates a persistent storage for food embeddings.
Loads the Sentence Transformer model for embedding generation.
Stores food embeddings in ChromaDB.
Asks the user for a PDF file containing a recipe.
Extracts text from the PDF.
Extracts ingredients from the text using regex.
Converts the ingredients into an embedding.
Queries ChromaDB for the top 5 similar recipes based on cosine similarity.
Displays the recommended recipes.
'''
def main():
    """Main function to extract, generate embeddings, and recommend food."""
    # Load food dataset
    with open("FoodDataSet.json", "r") as file:
        food_items = json.load(file)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("recipe_food")
    
    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Store food embeddings
    store_embeddings_in_chromadb(food_items, collection, model)
    
    # Extract and process recipe PDF
    file_path = input("Enter the path to the recipe PDF: ")
    text = extract_text_from_pdf(file_path)
    ingredients = extract_ingredients(text)
    
    if ingredients:
        print("Extracted Ingredients:", ingredients)
        recipe_embedding = generate_embeddings(" ".join(ingredients), model)
        
        # Query ChromaDB for similar recipes
        results = collection.query(query_embeddings=[recipe_embedding], n_results=5)
        
        if results["ids"] and results["ids"][0]:
            print("Recommended Recipes:")
            for index, food_id in enumerate(results["ids"][0]):
                recommended_item = food_items[int(food_id)]
                print(f"Top {index + 1} Recommended Item ==> {recommended_item['food_name']}")
        else:
            print("No similar recipes found.")
    else:
        print("No ingredients found in the recipe.")

if __name__ == "__main__":
    main()

'''
 Summary of the Workflow
1️⃣ Load Food Dataset → Reads FoodDataSet.json.
2️⃣ Initialize ChromaDB → Creates a vector database for food items.
3️⃣ Generate Food Embeddings → Converts food ingredients into vector embeddings.
4️⃣ Store Embeddings in ChromaDB → Saves food embeddings for similarity search.
5️⃣ Extract Recipe from PDF → Reads a user-provided recipe and extracts text.
6️⃣ Extract Ingredients → Identifies ingredients using regex.
7️⃣ Generate Recipe Embeddings → Converts extracted ingredients into embeddings.
8️⃣ Find Similar Recipes → Queries ChromaDB using cosine similarity.
9️⃣ Display Recommended Recipes → Shows the top 5 similar recipes.
'''