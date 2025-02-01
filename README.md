# Food Recommendation System with Similarity Search using ChromaDB

This project is a food recommendation system that uses Natural Language Processing (NLP) and vector similarity search to suggest food items based on a user's query. The system leverages **ChromaDB** for vector storage, **SentenceTransformer** for generating text embeddings, and (optionally) **zero-shot classification using BART MNLI** for dietary categorization.

## Project Overview

This food recommendation system does the following:

1. **Classify User Query**: Classifies the user's query (e.g., "I want to eat curry for dinner") into one of the dietary categories: vegan, vegetarian, non-vegan, non-vegetarian, or pescatarian.
2. **Generate Embeddings**: Generates vector embeddings for a list of food items (with names and descriptions) using the **SentenceTransformer** model.
3. **Store Embeddings in ChromaDB**: Embeddings are stored in **ChromaDB**, a vector database, for efficient retrieval.
4. **Similarity Search**: Performs a similarity search to recommend the top 5 food items that are most similar to the user’s query based on the cosine similarity between embeddings.

## Requirements

Before running this project, you need to install the following dependencies:

- Python 3.x
- `transformers`
- `sentence-transformers`
- `chromadb`
- `torch`
- `json`

You can install these dependencies using pip:

```bash
pip install transformers sentence-transformers chromadb torch
```

## Project Files

- `foodDataSet.json`: A JSON file that contains a list of food items, each with a unique ID, name, and description.
- `foodRecommendationSystem.py`: The main script that loads the food data, classifies user input, generates embeddings, stores data in ChromaDB, and performs similarity search.

## Project Workflow

### 1. Loading Food Data

The project loads a list of food items from a `FoodDataSet.json` file. 

### 2. User Query Classification

The user can input a query (e.g., "I want to eat curry for dinner"). This query is classified into a dietary category using Hugging Face’s **BART model** with zero-shot classification. The possible categories include:

- Vegan
- Vegetarian
- Non-vegan
- Non-vegetarian
- Pescatarian

### 3. Embedding Generation

The system uses the **SentenceTransformer** (`all-MiniLM-L6-v2`) model to generate vector embeddings for both food item descriptions and the user’s query. The embeddings are stored in **ChromaDB**.

### 4. Storing Data in ChromaDB

The embeddings and their corresponding food items are added to **ChromaDB**. The database ensures that no duplicates are stored by checking for existing food IDs.

### 5. Similarity Search

The system performs a similarity search based on the query embedding and retrieves the top 5 most similar food items from the database. The similarity is calculated using **cosine similarity**, and the most relevant food items are returned.

### 6. Displaying Recommendations

The system displays the top 5 food recommendations, including their names, similarity scores, and descriptions.

## How to Run

1. Make sure you have the required dependencies installed.
2. Place the `foodDataSet.json` file with food item data in the same directory.
3. Run the `foodRecommendationSystem.py` script:

```bash
python main.py
```

4. Input a query (e.g., "I want to eat curry for dinner") when prompted, and the system will display the top 5 recommended food items based on similarity.

## Example Output

```
Classification Result:
  - Non-vegan: 0.92
  - Non-vegetarian: 0.88
  - Vegan: 0.62
  - Vegetarian: 0.55
  - Pescatarian: 0.53

==================================================

Top 5 Recommended Food Items Based on Your Query:
1. Vegetable Curry
   Score: 0.8563
   Description: A spicy vegetable curry with a mix of seasonal vegetables.
--------------------------------------------------
2. Chicken Tikka
   Score: 0.8479
   Description: Grilled marinated chicken with traditional spices.
--------------------------------------------------
```

## Technologies Used

- **Hugging Face**: For the **BART** model (zero-shot classification).
- **Sentence Transformers**: For generating text embeddings using **all-MiniLM-L6-v2**.
- **ChromaDB**: For storing and querying high-dimensional vectors (embeddings).
- **Python**: For implementation of the logic.

  ## Next Steps
-**Personalized User Experience**
By classifying user queries into different categories, you could create a more personalized experience for the user. This would allow you to suggest not just similar food items, but diet-specific meals that align with the user’s preferences.
How This Helps:
User Profiles: Over time, you can store user preferences (e.g., vegan, vegetarian) and combine them with classification results for dynamic personalization of recommendations.
Contextual Awareness: The system will be aware of user preferences and will adapt the recommendations to offer only those food items that fit within those preferences.

-**Improved Search Efficiency**
If you know that the user’s query falls under a particular dietary category, you could optimize the search to only look for food items within that category. This reduces the number of embeddings to compare and can improve the efficiency of the search.
How This Helps:
Faster Searches: When you classify the query, you can pre-select a subset of the database that matches the category, which leads to faster searches by reducing the amount of data the system needs to process.

