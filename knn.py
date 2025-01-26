from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
import numpy as np
import string
from nltk.stem import PorterStemmer
import pandas as pd

train_path = '/kaggle/input/vscp-pml-unibuc-2024/train.csv'
valid_path = '/kaggle/input/vscp-pml-unibuc-2024/val.csv'
test_path = '/kaggle/input/vscp-pml-unibuc-2024/test.csv'

# load data
train_data = pd.read_csv(train_path)
valid_data = pd.read_csv(valid_path)
test_data = pd.read_csv(test_path)

# extract texts and scores
train_texts = train_data['text'].tolist()
valid_texts = valid_data['text'].tolist()
train_scores = train_data['score'].values
valid_scores = valid_data['score'].values

# check how much train data we have before augmentation
print(len(train_texts))

# duplicate entries where the score is neither 1 nor 0
filtered_data = [(train_texts[i], train_scores[i]) for i in range(len(train_texts)) if train_scores[i] != 1 and train_scores[i] != 0]
train_texts += [x[0] for x in filtered_data]
train_scores = np.concatenate((train_scores, [x[1] for x in filtered_data]))

# check how much train data we have after augmentation
print(len(train_texts))



# apply stemming
stemmer = PorterStemmer()
train_texts = [" ".join([stemmer.stem(word) for word in text.split()]) for text in train_texts]
valid_texts = [" ".join([stemmer.stem(word) for word in text.split()]) for text in valid_texts]

# remove punctuation (characters that are neither space, nor alpha-numeric)
train_texts = [''.join([char for char in text if char.isalnum() or char.isspace()]) for text in train_texts]
valid_texts = [''.join([char for char in text if char.isalnum() or char.isspace()]) for text in valid_texts]

# convert to lowercase
train_texts = [text.lower() for text in train_texts]
valid_texts = [text.lower() for text in valid_texts]

# build a TF-IDF vectorizer based on fine-tuned parameters
tfidf = TfidfVectorizer(
    max_features=7000,
    max_df=0.85,
    min_df=2,
    ngram_range=(1, 2),
    sublinear_tf=True,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    stop_words='english'
)

# vectorize train and validation data
X_train = tfidf.fit_transform(train_texts)
X_valid = tfidf.transform(valid_texts)


knn = KNeighborsRegressor(
    n_neighbors=119,
    metric='cosine',
    weights='distance'
)

# train the model
knn.fit(X_train, train_scores)

# evaluate the model on validation data
y_pred = knn.predict(X_valid)
spearman_corr = spearmanr(valid_scores, y_pred)

print(f"Spearman on validation: {spearman_corr.correlation}")


# combine train data with validation data and retrain the model
combined_texts = np.concatenate([train_texts, valid_texts])
combined_scores = np.concatenate([train_scores, valid_scores])

X_combined = tfidf.fit_transform(combined_texts)

final_model = knn.fit(X_combined, combined_scores)


# preprocess and vectorize test data
test_texts = test_data['text'].tolist()
test_texts = [" ".join([stemmer.stem(word) for word in text.split()]) for text in test_texts]
test_texts = [''.join([char for char in text if char.isalnum() or char.isspace()]) for text in test_texts]
test_texts = [text.lower() for text in test_texts]
test_features = tfidf.transform(test_texts) 

# make predictions for test data
test_predictions = final_model.predict(test_features)

# save predictions
output_path = '/kaggle/working/predictions_day6_2.csv'
output_df = pd.DataFrame({
    'id': test_data['id'],
    'score': test_predictions
})

output_df.to_csv(output_path, index=False)