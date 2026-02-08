import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/kaggle/dataset/imdb dataset.csv")
df.head()
df['sentiment']=df['sentiment'].map({'positive':1,'negative':0})
X=df['review']
y=df['sentiment']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

tfidf=TfidfVectorizer(stop_words="english",max_features=5000)

X_train_tfidf=tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)

nb_model=MultinomialNB()
nb_model.fit(X_train_tfidf,y_train)
y_pred=nb_model.predict(X_test_tfidf)
accuracy=accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)

print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
def predict_sentiment(review):
  review_tfidf=tfidf.transform([review])
  prediction=nb_model.predict(review_tfidf)
  return "Positive" if prediction[0]==1 else "Negative"
predict_sentiment("The movie was boring and a complete waste of time")