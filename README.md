Sai Gulve 202201040144

Siddhi Bambal 202201040127

Dhanvantari Chauvan 202201040143

📚 SMS Spam Detection - NLP Preprocessing and Text Classification
🔹 Objective:
The goal of this assignment is to apply Natural Language Processing (NLP) techniques to preprocess SMS text messages and build a machine learning model to classify them as Spam or Ham (not spam).

🔹 Dataset:
Dataset Source: SMS Spam Collection

Description: A collection of labeled SMS messages — ham (legitimate) or spam (unwanted).

🔹 Tasks Performed:
✅ NLP Preprocessing:

Lowercasing text

Tokenization

Removal of stopwords

Stemming (Porter Stemmer)

Lemmatization (WordNet Lemmatizer)

✅ Feature Extraction:

CountVectorizer (simple word frequency)

TF-IDF Vectorizer (importance based)

✅ Model Development:

Machine Learning Algorithm: Logistic Regression

✅ Model Evaluation:

Metrics: Accuracy, Precision, Recall, F1-Score

Visualization: Confusion Matrix Heatmaps

✅ Analysis:

Compared CountVectorizer and TF-IDF Vectorizer performance.

Discussed model strengths and further improvement ideas.

🔹 Key Results:

Vectorizer	Accuracy	Precision	Recall	F1-Score
CountVectorizer	~98%	High	High	High
TF-IDF Vectorizer	~98.5%	High	High	High
✅ TF-IDF performed slightly better than CountVectorizer in terms of model evaluation metrics.

🔹 Conclusion:
NLP preprocessing significantly improves text classification performance.

Logistic Regression is an effective baseline for spam detection tasks.

Using TF-IDF representation yields better results compared to simple CountVectorizer.

Future work could involve testing deep learning models like LSTM or CNN for better results.

🔹 Tools and Libraries Used:
Python

Pandas, NumPy

NLTK (Natural Language Toolkit)

Scikit-learn (sklearn)

Seaborn, Matplotlib

🔹 How to Run:
Open the Colab Notebook (.ipynb file) provided.

Run all cells sequentially.

Observe preprocessing steps, model training, evaluation, and analysis.

Modify or experiment if desired for extra exploration!
