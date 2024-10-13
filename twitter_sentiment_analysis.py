# Before we begin, we supress deprecation warnings resulting from nltk on Kaggle
import warnings
import collections
import pandas as pd
import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

warnings.filterwarnings("ignore", category=DeprecationWarning)


def EDA_and_load_data(filename):
    tweets_df = pd.read_csv("Tweets.csv")
    print(tweets_df.columns.values)
    print(tweets_df.head())

    sentiment_counts = tweets_df['airline_sentiment'].value_counts()
    num_of_tweets = tweets_df["tweet_id"].count()
    print(sentiment_counts)
    print(num_of_tweets)
    return tweets_df


def normalizer(tweet):
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
    only_letters = re.sub("[^a-zA-Z]", " ", tweet)
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


def main():
    # Load Data
    fn = "Tweets.csv"
    tweets_df = EDA_and_load_data(fn)

    # Sample use of normalizer function
    sample_text = "Throughout all of heaven and earth, I alone am the honored one."
    lemmas = normalizer(sample_text)
    print(lemmas)

    # Running the Normalizer function on the text column of the data
    pd.set_option('display.max_colwidth', -1)  # Setting this allows us to see full content of the cells
    tweets_df['normalized_tweets'] = tweets_df.text.apply(normalizer)
    print(tweets_df[['text', 'normalized_tweets']].head())

    # Generate Bigrams and Trigrams:
    grams_list = []
    for tweet in tweets_df['normalized_tweets']:
        bigram = list(ngrams(tweet, 2))
        trigram = list(ngrams(tweet, 3))
        gram = bigram + trigram
        grams_list.append(gram)

    tweets_df['grams'] = grams_list
    print(tweets_df[['normalized_tweets', 'grams']])

    # Some Counting to get some extra insight into the data:

    print(tweets_df[(tweets_df.airline_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(20))
    # Sentences like "cancelled flight", "late flight", "booking problems", "delayed flight" stand out clearly.

    print(tweets_df[(tweets_df.airline_sentiment == 'positive')][['grams']].apply(count_words)['grams'].most_common(20))
    # Mostly Bigrams of 'customer service' and other related texts










    # Data Preparation: Train Test Split:
    x_train, x_test, y_train, y_test = train_test_split(tweets_df['text'], tweets_df['airline_sentiment'],
                                                        test_size=0.3, random_state=42, shuffle=True)

    # Data Preparation: Vectorization:
    cv = CountVectorizer(ngram_range=(1, 2), max_features=10000)
    cv.fit(x_train)
    x_train = cv.transform(x_train).toarray()
    x_test = cv.transform(x_test).toarray()
    print("Vectorization Complete")

    # Data Preparation: Label Encoding:
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    print("Label Encoding Complete")

    # Model Building:

    # SVM:
    clf = svm.SVC(kernel='linear', class_weight="balanced")
    clf.fit(x_train, y_train)
    print("Fitting Complete")
    y_pred = clf.predict(x_test)
    print('Prediction Complete')
    acc_score = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy{acc_score}")

    # MNB:
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_pred_mnb = mnb.predict(x_test)
    acc_score_mnb = accuracy_score(y_test, y_pred_mnb)
    print(f"MNB Score: {acc_score_mnb}")

    # CNB:
    cnb = ComplementNB()
    cnb.fit(x_train, y_train)
    y_pred_cnb = cnb.predict(x_test)
    acc_score_cnb = accuracy_score(y_test, y_pred_cnb)
    print(f"CNB Score: {acc_score_cnb}")

    # GNB:
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred_gnb = gnb.predict(x_test)
    acc_score_gnb = accuracy_score(y_test, y_pred_gnb)
    print(f"GNB Score: {acc_score_gnb}")


if __name__ == '__main__':
    main()
