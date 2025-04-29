import nltk
nltk.download('punkt')

from textblob import TextBlob
from newspaper import Article


url = 'https://www.ndtv.com/world-news/israel-hamas-war-gaza-death-count-crosses-25-000-says-hamas-as-israel-intensifies-offensive-4904309'
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary

# # using text file
# with open('mytext.txt','r') as f:
#     text = f.read()

blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)
