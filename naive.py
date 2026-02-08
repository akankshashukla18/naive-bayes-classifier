import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="IMDB Sentiment Analysis")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Naive Bayes + TF-IDF based sentiment prediction")

# -----------------------------
# Create dataset inside code
# -----------------------------
@st.cache_data
def load_data():
    data = {
        "review": [
            "The movie was fantastic and full of emotions",
            "Absolutely loved the acting and the story",
            "One of the best movies I have ever seen",
            "The film was boring and too long",
            "Worst movie, complete waste of time",
            "Terrible acting and bad storyline",
            "Amazing direction and great performance",
            "I did not like the movie at all",
            "The plot was weak and disappointing",
            "A wonderful and inspiring movie"
        ],
        "sentiment": [
            1, 1, 1, 0, 0, 0, 1, 0, 0, 1
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(stop_words="english")

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    return model, tfidf, y_test, y_pred

model, tfidf, y_test, y_pred = train_model(df)

# -----------------------------
# Model performance
# -----------------------------
st.subheader("📊 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.2f}")

if st.checkbox("Show Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

if st.checkbox("Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# -----------------------------
# Prediction function
# -----------------------------
def predict_sentiment(review):
    review_tfidf = tfidf.transform([review])
    prediction = model.predict(review_tfidf)
    return "Positive 😊" if prediction[0] == 1 else "Negative 😞"

# -----------------------------
# User input
# -----------------------------
st.subheader("✍️ Enter a Movie Review")

user_review = st.text_area(
    "Type your review:",
    height=120,
    placeholder="The movie was boring and a complete waste of time"
)

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = predict_sentiment(user_review)
        st.success(f"**Predicted Sentiment:** {result}")




