[README.md](https://github.com/user-attachments/files/21590487/README.md)

"This project performs sentiment analysis on Flipkart product reviews using multiple approaches — including traditional machine learning algorithms (Logistic Regression, Naïve Bayes, Support Vector Machines, Random Forest, etc.) and advanced deep learning techniques like BERT. It compares the performance of these methods to classify reviews as positive, negative, or neutral, providing insights into the most effective model for text sentiment classification."


Sentiment analysis in python using different techniques:
VADER (Valence Aware Dictionary and sEntiment Reasoner) - Bag of words approach

Roberta Pretrained Model from 🤗

Huggingface Pipeline

TRADITIONAL ML ALGO

# **Read in Data and NLTK**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
```

```python
# Read in data
df = pd.read_csv('flipkart_data.csv')

```

```python
df.head()
```

```python
df.shape
```

```python
df=df.head(1000)
```

# EDA

```python
# Remove duplicates
df.drop_duplicates(inplace=True)
```

```python
# Create a sentiment column based on rating
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Encode sentiment labels (this step is now redundant as we created the column directly)
# df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Display the first few rows with the new sentiment column
display(df.head())
```

```python
df.shape
```

```python
ax = df['rating'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()
```

```python

```

```python

```

```python
df['review'][50]
```

```python
import re
import string
```

```python
# Clean the review text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()
```

```python
# Apply cleaning
df['review'] = df['review'].apply(clean_text)
```

```python
df['review']
```

# Basic NLTK

```python
nltk.download('punkt_tab')
```

```python
nltk.download('averaged_perceptron_tagger_eng')
```

```python
nltk.download('maxent_ne_chunker_tab')
```

```python
example = df['review'][50]
print(example)
```

```python
tokens = nltk.word_tokenize(example)
tokens[:20]
```

```python
tagged = nltk.pos_tag(tokens)
tagged[:20]
```

```python
nltk.download('words')
```

```python
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
```

# **VADER Seniment Scoring**

I will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.



This uses a "bag of words" approach:

Stop words are removed

each word is scored and combined to a total score.

```python
nltk.download('vader_lexicon')
```

```python
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
```

```python
sia.polarity_scores('I am so happy!')
```

```python
sia.polarity_scores('This is the worst thing ever.')
```

```python
sia.polarity_scores(example)
```

# Run the polarity score on the entire dataset

```python
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['review']
    res[i] = sia.polarity_scores(text)
```

```python
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
# Include the 'sentiment' and 'rating' columns from df during the merge
vaders = vaders.merge(df[['sentiment', 'rating']], left_on='Id', right_index=True)
```

```python
# sentiment score and metadata
vaders.head()
```

```python
df.head()
```

```python
# Showing target variable counts
print(df['sentiment'].value_counts())
```

# Plot VADER results

```python
ax = sns.barplot(data=vaders, x='rating', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()
```

```python
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='rating', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='rating', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='rating', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Negative')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()
```

Interpretation:

VADER Sentiment Scores Are Valid.



The sentiment scores align well with star ratings, validating the sentiment model.

# Roberta Pretrained Model

---> Use a model trained of a large corpus of data.



---> Transformer model accounts for the words but also the context related to other words.

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
```

```python
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
```

```python
# VADER results on example
print(example)
sia.polarity_scores(example)
```

```python
# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)
```

```python
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict
```

```python
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['review']
        # Removed the line accessing 'Id' as it's not in the DataFrame
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[i] = both # Use the index 'i' as the key
    except RuntimeError:
        print(f'Broke for row index {i}') # Print row index instead of 'id'
```

```python
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left', left_on='Id', right_index=True)
```

# Compare Scores between models

```python
results_df.columns
```

# **Combine and compare**

```python
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='sentiment',
            palette='tab10')
plt.show()
```

# Review Examples:





```python
results_df.query('sentiment == 1') \
    .sort_values('roberta_pos', ascending=False)['review'].values[0]
```

```python
results_df.query('sentiment == 1') \
    .sort_values('vader_pos', ascending=False)['review'].values[0]
```

```python
results_df.query('rating == 5') \
    .sort_values('roberta_neg', ascending=False)['review'].values[0]
```

```python
results_df.query('rating == 5') \
    .sort_values('vader_neg', ascending=False)['review'].values[0]
```

```python

```

# **The Transformers Pipeline**

```python
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")
```

```python
sent_pipeline('I do love sentiment analysis!')
```

```python
sent_pipeline('Make sure to like and subscribe!')
```

```python
sent_pipeline('booo')
```

```python
sent_pipeline("i don't like this")
```

```python
sent_pipeline('lol!what it is?')
```

```python
sent_pipeline('Good quality product. But the side fixable rings are not fitted properly and the black rubber frame should have been more thicker and sturdy.')
```

```python

```

```python

```
#SENTIMENT ANALYSIS WITH ML ALGO

# **1. Importing Necessary Libraries and Dataset:**

• Load required Python libraries:



o Pandas for handling datasets.



o Scikit-learn for machine learning algorithms and vectorization.



o Matplotlib/Seaborn for data visualization.



o WordCloud for visualizing common words in reviews.



o Warnings to suppress unnecessary messages.



• Load the dataset using Pandas and explore its structure.

```python
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')  # Suppresses unnecessary warnings
```

```python
# Load the dataset
df = pd.read_csv('flipkart_data.csv')
```

```python
# Display the first few rows
df.head()
```

```python
df.shape
```

```python

```

# **2: Data Preprocessing**

**• Remove missing values and duplicate entries.**

```python

```

```python
# Check for missing values
df.isnull().sum()
```

```python
# Drop missing values
df.dropna(inplace=True)
```

```python

# Remove duplicates
df.drop_duplicates(inplace=True)
```

**Convert text into lowercase and remove stopwords, punctuation, and

special characters**

```python
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
```

```python
# Clean the review text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()
```

```python
# Apply cleaning
df['review'] = df['review'].apply(clean_text)
```

```python
# Create a sentiment column based on rating
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 3 else 0)

# Encode sentiment labels (this step is now redundant as we created the column directly)
# df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Display the first few rows with the new sentiment column
display(df.head())
```

```python
df['sentiment'].value_counts()
```

```python
# Visualize the sentiment distribution using a pie chart
sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=['Positive', 'Negative'], autopct='%1.1f%%', startangle=90, colors=['yellow', 'salmon'])
plt.title('Sentiment Distribution')
plt.show()
```

```python
# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['review']).toarray()
y = df['sentiment']
```

```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
X_test
```

```python
X_train
```

# **3. Exploratory Data Analysis (EDA):**

**Visualize sentiment distribution using count plots.**

```python
# Sentiment distribution
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()
```

**• Create a word cloud to identify common words in positive and negative

reviews.**

```python
# WordCloud for positive reviews
pos_text = ' '.join(df[df['sentiment'] == 1]['review'])
wordcloud_pos = WordCloud(width=800, height=500).generate(pos_text)
plt.imshow(wordcloud_pos)
plt.axis('off')
plt.title('Positive Reviews WordCloud')
plt.show()
```

```python
# WordCloud for negative reviews
neg_text = ' '.join(df[df['sentiment'] == 0]['review'])
wordcloud_neg = WordCloud(width=800, height=500).generate(neg_text)
plt.imshow(wordcloud_neg)
plt.axis('off')
plt.title('Negative Reviews WordCloud')
plt.show()
```

**• Analyze correlations between review length and sentiment.**

```python
# Review length vs sentiment
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
sns.boxplot(x='sentiment', y='review_length', data=df)
plt.title('Review Length by Sentiment')
plt.show()
```

 Explanation:

Visualizes sentiment distribution.



Word clouds highlight frequent words in each sentiment.



Boxplot shows review length correlation with sentiment.

```python

```

# **4. Model Training and Selection:**

• Train different machine learning models:  



o Logistic Regression



o Naïve Bayes



o Random Forest Classifier



o Support Vector Machine (SVM)







• Compare model performance using accuracy and F1-score.




```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'SVM': SVC(class_weight='balanced')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}')

```

```python
# Create a DataFrame to store model performance metrics
performance_data = {'Model': models.keys(),
                    'Accuracy': [accuracy_score(y_test, model.predict(X_test)) for model in models.values()],
                    'F1 Score': [f1_score(y_test, model.predict(X_test)) for model in models.values()]}
performance_df = pd.DataFrame(performance_data)

# Plot the performance metrics
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(performance_df['Model']))

plt.bar(index, performance_df['Accuracy'], bar_width, label='Accuracy', color='skyblue')
plt.bar(index + bar_width, performance_df['F1 Score'], bar_width, label='F1 Score', color='lightgreen')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width / 2, performance_df['Model'])
plt.legend()
plt.ylim(0.8, 1) # Set y-axis limits for better visualization of differences
plt.tight_layout()
plt.show()
```



# **5. Model Evaluation and Prediction:**

• Evaluate the best model using:  



o Accuracy Score



o Precision, Recall, F1-Score



o Confusion Matrix

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression # Import LogisticRegression

# Train best model (Logistic Regression in this case)
best_model = LogisticRegression(class_weight='balanced', max_iter=1000) # Use LogisticRegression
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Evaluation
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
# Compare training and test results
comparison_data = []

for name, model in models.items():
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    comparison_data.append([name, train_acc, test_acc, train_f1, test_f1])

comparison_df = pd.DataFrame(comparison_data, columns=['Model', 'Train Accuracy', 'Test Accuracy', 'Train F1 Score', 'Test F1 Score'])
display(comparison_df)
```

```python
# Plot the comparison of training and test results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

bar_width = 0.35
index = np.arange(len(comparison_df['Model']))

# Plot Accuracy
axes[0].bar(index, comparison_df['Train Accuracy'], bar_width, label='Train Accuracy', color='skyblue')
axes[0].bar(index + bar_width, comparison_df['Test Accuracy'], bar_width, label='Test Accuracy', color='steelblue')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Training vs. Test Accuracy Comparison')
axes[0].set_xticks(index + bar_width / 2)
axes[0].set_xticklabels(comparison_df['Model'])
axes[0].legend()
axes[0].set_ylim(0.8, 1.05)

# Plot F1 Score
axes[1].bar(index, comparison_df['Train F1 Score'], bar_width, label='Train F1 Score', color='lightgreen')
axes[1].bar(index + bar_width, comparison_df['Test F1 Score'], bar_width, label='Test F1 Score', color='forestgreen')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Training vs. Test F1 Score Comparison')
axes[1].set_xticks(index + bar_width / 2)
axes[1].set_xticklabels(comparison_df['Model'])
axes[1].legend()
axes[1].set_ylim(0.8, 1.05)

plt.tight_layout()
plt.show()
```

Test the model on new reviews to classify sentiment as positive or negative.

# **Predict on New Data**

```python
# Predict sentiment of new reviews
new_reviews = ["This phone is amazing!", "Terrible product. Waste of money."]
cleaned = [clean_text(review) for review in new_reviews]
vectorized = tfidf.transform(cleaned).toarray()
predictions = best_model.predict(vectorized)
for review, sentiment in zip(new_reviews, predictions):
    print(f'Review: "{review}" => Sentiment: {"Positive" if sentiment else "Negative"}')

```

```python
# Predict sentiment of new reviews
new_reviews = ["The product is POOR","Not ABLE sign in on Netflix and many other apps","Absolutely loved it, works perfectly fine!"]
cleaned = [clean_text(review) for review in new_reviews]
vectorized = tfidf.transform(cleaned).toarray()
predictions = best_model.predict(vectorized)
for review, sentiment in zip(new_reviews, predictions):
    print(f'Review: "{review}" => Sentiment: {"Positive" if sentiment else "Negative"}')

```
