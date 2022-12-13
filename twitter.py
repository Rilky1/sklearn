from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
import string
import re
from wordcloud import WordCloud
import random


def readFile():
    """
    Purpose: read in the CSV file and change positive sentiment tweets from 4 to 1
    Parameters: None
    Returns: read in data file
    """

    columns = ["sentiment", "id", "date", "query", "user_id", "text"]
    df = pd.read_csv("../input/training.1600000.processed.noemoticon.csv", encoding="latin", names=columns)
    df.head()

    # replacing positive = 4 to 1
    df["sentiment"] = df["sentiment"].replace(4,1)
   
    return df 


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    """
    Purpose: Plot a bar graph displaying the number of tweets with sentiment 0 and number of tweets with sentiment 1
    Parameters: Dataset, number of graphs being shown & number of graphs per row (formatting) 
    Returns: None/Plots graph
    """
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int((nCol + nGraphPerRow - 1) / nGraphPerRow)
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    """
    Purpose: 
    Parameters:
    Returns:
    """
    filename = pd.DataFrame(df)
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


def posWordCloud(df):
    """
    Purpose: In the positive tweets, generate a word cloud that tells us the most used words in positive tweets
    Parameters: Dataset
    Returns: None/Plots wordcloud
    """
    # creating a string of positive tweets to analyze the words
    positive_tweets = df[df['sentiment'] == 1]['text'].tolist()
    positive_tweets_string = " ".join(positive_tweets)
    plt.imshow(WordCloud().generate(positive_tweets_string))
    plt.axis("off")
    plt.show()
    # positive words: love, thank, lol, miss


def negWordCloud(df):
    """
    Purpose: In the negative tweets, generate a word cloud that tells us the most used words in negative tweets.
    Parameters: Dataset
    Returns: None/Plots wordcloud
    """
    # creating the string of negative tweets to analyze the words
    negative_tweets = df[df['sentiment'] == 0]['text'].tolist()
    negative_tweets_string = " ".join(negative_tweets)
    plt.imshow(WordCloud().generate(negative_tweets_string))
    plt.axis("off")
    plt.show()
    # negative words: work, want, now, today, going


def tweetsCleaner(tweet):
    """
    Purpose: For each tweet, remove unecessary text, including urls, numbers, tags, punctuation, and stop words
    Parameters: String of a tweet
    Returns: Modified tweet in a list contatining remaining words
    """

    # removing the urls 
    tweet = re.sub(r'((www.\S+)|(https?://\S+))', r"", tweet)
    #removing the numbers 
    tweet = re.sub(r'[0-9]\S+', r'', tweet)
    #removing the tags 
    tweet = re.sub(r'(@\S+) | (#\S+)', r'', tweet)
    
    # removing the punctuation 
    tweet_without_punctuation = []
    for char in tweet:
        if char not in string.punctuation:
            tweet_without_punctuation.append(char)
    tweet_without_punctuation = "".join(tweet_without_punctuation)
   
    # removing the stop words
    stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    tweet_without_stopwords = []
    for word in tweet_without_punctuation.split():
        if word.lower() not in stop_words:
            tweet_without_stopwords.append(word)
    
    return tweet_without_stopwords


def vectorizer(df):
    """
    Purpose: Reformat the dataset by calling in tweetsCleaner and removing fluff words, and then converting dataset into numbers readable by the computer
    Parameters: Dataset 
    Returns: Reformatted dataset in a matrix of numbers ready for training and testing
    """

    # extract the features using count vectorizer
    vectorizer = CountVectorizer(analyzer = tweetsCleaner, dtype = 'uint8')
    df_countvectorizer = vectorizer.fit_transform(df['text'])
    df_countvectorizer.shape

    return df_countvectorizer


def trainData(df, df_countvectorizer):
    """
    Purpose: Split dataset into train test split, and then given the classifier to use, train and test the data and 
    determine the accuracy of the classifier on determining sentiments.
    Parameters: Dataset & reformatted dataset.
    Returns: None
    """

    while True:
        # splitting the features into train and test
        X_train, X_test, y_train, y_test = train_test_split(df_countvectorizer, df['sentiment'], test_size = 0.2, shuffle = True)
        
        user_input = str(input("Choose a classifier (MNB, LR, or LSVC): "))
        if user_input.lower() == "multinomialnb" or user_input.lower() == "mnb":
            classifier = MultinomialNB()
        elif user_input.lower() == "logistic regression" or user_input.lower() == "lr":
            classifier = LogisticRegression()
        elif user_input.lower() == "lsvc":
            classifier = LinearSVC()
        else:
            break
   
        classifier.fit(X_train, y_train)

        pred = classifier.predict(X_test)

        print(classification_report(y_test, pred))

    return y_test, pred


def samplePred(y_test, pred):
    """
    Purpose: Take a snippet of tweets and compare predicted model sentiment with actual sentiment 
    Parameters: actual sentiment array from test segment, prediction list
    Returns: None (prints values and tweets)
    """

    y_test_list = []
    for i in y_test:
        y_test_list.append(i)

    num = random.randint(0, len(pred) - 5)
    for i in range(num, num+5):
        print()
        print("Model Pred:", pred[i])
        print("Actual:", y_test_list[i])
  

def main():

    df = readFile()
    # plotPerColumnDistribution(df, 10, 5)
    # plotCorrelationMatrix(df, 8)
    # plotScatterMatrix(df, 10, 5)
    # posWordCloud(df)
    # negWordCloud(df)
    df_countvectorizer = vectorizer(df)
    y_test, pred = trainData(df, df_countvectorizer)
    samplePred(y_test, pred)

main()