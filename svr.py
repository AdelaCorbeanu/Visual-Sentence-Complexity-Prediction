import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, kendalltau

train_path = '/kaggle/input/vscp-pml-unibuc-2024/train.csv'
valid_path = '/kaggle/input/vscp-pml-unibuc-2024/val.csv'
test_path = '/kaggle/input/vscp-pml-unibuc-2024/test.csv'

# load the data
train_data = pd.read_csv(train_path)
valid_data = pd.read_csv(valid_path)
test_data = pd.read_csv(test_path)

# extract texts
train_texts = train_data['text'].tolist()
valid_texts = valid_data['text'].tolist()

# extract scores
train_scores = train_data['score'].values
valid_scores = valid_data['score'].values

vectorizer = TfidfVectorizer()

# vectorize data
X_train = vectorizer.fit_transform(train_texts)
X_valid = vectorizer.transform(valid_texts)

# build SVR model with best parameters according to experiments
svr_model = SVR(kernel='rbf', C=1, gamma=1)

# train the model
svr_model.fit(X_train, train_scores)

# make predictions for validation data
valid_predictions = svr_model.predict(X_valid)

# evaluate the model on validation data using Spearman correlation
spearman, _ = spearmanr(valid_scores, valid_predictions)
print(f'Spearman on validation: {spearman}')

# extract test data and vectorize it
test_texts = test_data['text'].tolist()
X_test = vectorizer.transform(test_texts)

# make predictions for test data
test_predictions = svr_model.predict(X_test)

# save the test data predictions
test_data['score'] = test_predictions
test_data[['id', 'score']].to_csv('predictions_day4_3.csv', index=False)