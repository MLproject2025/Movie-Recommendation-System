
# 🎬 CineMatch: Hybrid Movie Recommendation System

CineMatch is a web-based hybrid movie recommendation engine built with **Streamlit**. It blends **Collaborative Filtering** (using user similarity) and **Content-Based Filtering** (using TF-IDF on movie titles) to suggest personalized movies to users. The system is powered by the open-source **MovieLens 100k** dataset.

This app was built as part of a machine learning project and is deployable on [Streamlit Cloud](https://streamlit.io/cloud).

---

## 🚀 Features

- ✅ Choose any User ID (1–943)
- 🎛️ Adjust the balance between collaborative and content-based filtering
- 📊 Control how many movie recommendations to view
- 📚 Uses cosine similarity for both users and movie title features
- 📈 Clean Streamlit UI with instant feedback

---

## 🧠 How It Works

- **Collaborative Filtering (CF):**  
  Measures similarity between users based on their movie ratings (user-user cosine similarity).

- **Content-Based Filtering (CBF):**  
  Uses **TF-IDF vectorization** on movie titles to calculate similarity between movies.

- **Hybrid Recommendation:**  
  Blends both scores with a configurable weight:
  Hybrid Score = (CF Weight × CF Score) + (1 - CF Weight × CB Score)

---

## 📂 Project Structure

```
📁 movie-recommender-streamlit/
├── streamlit_hybrid_recommender.py   # Main app
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
```

---

## 📦 Installation & Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_hybrid_recommender.py
```

---

## 📊 Dataset

- MovieLens 100k
- Source: https://grouplens.org/datasets/movielens/100k/

---

## ✅ Requirements

- pandas
- numpy
- scikit-learn
- streamlit

(Add to `requirements.txt`)

---

## 🙌 Acknowledgements

Thanks to [GroupLens Research](https://grouplens.org) for the MovieLens dataset and to the open-source community for making data science tools accessible.
