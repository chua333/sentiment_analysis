import pip as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


df = pd.read_csv('tweets.csv')

text = df['tweet_text']

target = df['is_there_an_emotion_directed_at_a_brand_or_product']

target = target[pd.notnull(text)]
text = text[pd.notnull(text)]

count_vect = CountVectorizer()
count_vect.fit(text)
counts = count_vect.transform(text)

clf = MultinomialNB()

clf.fit(counts, target)

print(clf.predict(count_vect.transform(['i hate my iphone'])))

predictions = clf.predict(counts[6000:9092])
correct_predictions = sum(predictions == target[6000:9092])
incorrect_predictions = (9092 - 6000) - correct_predictions

train_predictions = clf.predict(counts[0:6000])
train_correct_predictions = sum(train_predictions == target[0:6000])
train_incorrect_predictions = 6000 - train_correct_predictions

train_accuracy = train_correct_predictions/(train_correct_predictions+train_incorrect_predictions)
val_accuracy = correct_predictions/(correct_predictions+incorrect_predictions)

print({"val_accuracy": val_accuracy, "train_accuracy": train_accuracy})
