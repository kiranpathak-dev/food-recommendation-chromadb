# **Food Recommendation System using ChromaDB**

This project implements a food recommendation system using **natural language processing (NLP) techniques** and **vector similarity search**. It allows users to enter a query (e.g., "I want to eat curry for dinner") and retrieves the most relevant food items based on semantic similarity.  

The system leverages:  
✅ **Hugging Face's BART model** for classifying the user query into dietary categories.  
✅ **Sentence Transformers** for generating vector embeddings of food items.  
✅ **ChromaDB** for storing embeddings and performing **cosine similarity-based nearest neighbor search**.  

---

## **📌 Features**
🔹 Loads a list of food items from `FoodDataSet.json`.  
🔹 Classifies the user query into dietary categories (**vegan, vegetarian, non-vegan, non-vegetarian, pescatarian**).  
🔹 Converts food descriptions and user queries into **vector embeddings**.  
🔹 Stores and retrieves embeddings using **ChromaDB**.  
🔹 Performs **similarity search using cosine similarity** to find the most relevant food items.  
🔹 Returns the **top 5 most similar food items** based on the user’s query.  

---

## **📦 Requirements**
Ensure you have **Python 3.x** installed and install the required dependencies using:

```bash
pip install transformers sentence-transformers chromadb torch
```

---

## **🚀 How It Works**
### **1️⃣ Load Food Data**
The system reads a list of food items from `FoodDataSet.json`, each containing:
- `food_id`: Unique identifier.
- `food_name`: Name of the food.
- `food_description`: A short description of the dish.

---

### **2️⃣ Classify User Query**
User input (e.g., `"I want to eat curry for dinner"`) is classified into one of the following dietary categories using **Hugging Face's BART model**:  
✔️ Vegan  
✔️ Vegetarian  
✔️ Non-vegan  
✔️ Non-vegetarian  
✔️ Pescatarian  

---

### **3️⃣ Generate Text Embeddings**
The system uses **Sentence Transformer (`all-MiniLM-L6-v2`)** to convert:  
✅ **Food descriptions** into high-dimensional **vector embeddings**.  
✅ **User query** into an **embedding vector** for comparison.  

---

### **4️⃣ Store Embeddings in ChromaDB**
Food item embeddings are stored in **ChromaDB**, a high-performance vector database.  

- **Duplicates are avoided** by checking if an item already exists in the collection before adding it.

---

### **5️⃣ Similarity Search Using Cosine Similarity**
The system performs a **similarity search** using **cosine similarity-based nearest neighbor search**:  
✔ **Cosine similarity** measures how similar two vectors are, with a value close to **1 indicating high similarity** and a value near **0 indicating no similarity**.  
✔ **Nearest Neighbor Search (NNS)** finds the **top 5 most relevant** food items.  

#### **Mathematical Formula for Cosine Similarity**
\[
\text{cosine similarity} = \frac{A \cdot B}{||A|| ||B||}
\]
where:
- \( A \) and \( B \) are the vector embeddings of the food item and the query.
- \( ||A|| \) and \( ||B|| \) are their respective magnitudes.

---

### **6️⃣ Display Recommendations**
The system retrieves the **top 5 most similar** food items based on **cosine similarity**.

#### **Example Output**
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

---

## **🛠 How to Run the Project**
1️⃣ Ensure you have installed the dependencies.  
2️⃣ Place the `FoodDataSet.json` file in the same directory.  
3️⃣ Run the script:  

```bash
python foodRecommendationSystem.py
```

4️⃣ Enter a query when prompted, and the system will return the **top 5 most relevant food items**.

---

## **🔍 Technologies Used**
✅ **Hugging Face Transformers** – For BART-based query classification.  
✅ **Sentence Transformers (`all-MiniLM-L6-v2`)** – For text embedding generation.  
✅ **ChromaDB** – For **efficient similarity search** using **cosine similarity**.  
✅ **Python** – For implementing the recommendation system.  

---


  ## **Next Steps**
- **Personalized User Experience**
By classifying user queries into different categories, you could create a more personalized experience for the user. This would allow you to suggest not just similar food items, but diet-specific meals that align with the user’s preferences.
How This Helps:
User Profiles: Over time, you can store user preferences (e.g., vegan, vegetarian) and combine them with classification results for dynamic personalization of recommendations.
Contextual Awareness: The system will be aware of user preferences and will adapt the recommendations to offer only those food items that fit within those preferences.

- **Improved Search Efficiency**
If you know that the user’s query falls under a particular dietary category, you could optimize the search to only look for food items within that category. This reduces the number of embeddings to compare and can improve the efficiency of the search.
How This Helps:
Faster Searches: When you classify the query, you can pre-select a subset of the database that matches the category, which leads to faster searches by reducing the amount of data the system needs to process.

