# Import the required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import panda as pd


# Read data
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

# Text Data Features

def seperate_ingredients(data):
    text_data = data['ingredients'].apply(','.join)
    return text_data

train_text = seperate_ingredients(train)
test_text = seperate_ingredients(test)
target = train['cuisine']

# Feature Engineering

tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train)
X_test = tfidf.transform(test)
# If you fit() to your test data, you'd compute a new mean and variance for each feature.
# This would bias your model with information from the test data.
lb = LabelEncoder()
y = lb.fit_transform(target)

classifier = SVC(C=100, 
                 kernel = 'rbf', # kernel type
                 degree = 3, # default value
                 gamma = 1,
                 coef0 = 1,
                 shrinking = True,
                 tol =0.001,
                 probability=False,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter= -1,
                 decision_function_shape=None,
                 random_state=None


)
model = OneVsRestClassifier(classifier, n_jobs = 4)

model.fit(X,y)

y_test = model.predict(X_test)
y_pred = lb.inverse_transform(y_test)

test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)
