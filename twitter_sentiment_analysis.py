import tweepy
from textblob import TextBlob
import csv

consumer_key = 'smfAOQtDrXWl7DNpM78CGjok8'
consumer_secret = 'IRAWfmBky4u5WPZvM0LdVZVern3o0I3PhFvzgnlhCKoqFoRIR2'

access_token = '3682178712-mY6hvg9ZweVWJh2avQjSv2OUmiQLgwvlp908pgM'
access_token_secret = '3Ja3XHftvpdgrg7VWJu70hvtDm9VblOZ52kz4JGWvpEqV'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

csvFile = open('sentiment_analysis.csv','a')

csvWriter = csv.writer(csvFile)

#for tweet in tweepy.Cursor(api.search, q = "google",since = "2014-02-14", until = "2014-02-15", lang = "en").items():
#	csvWriter.writerow([tweet.created_at,tweet.text.encode('utf-8'),TextBlob(tweet.text).sentiment]) 
#	print tweet.created_at,tweet.text,TextBlob(tweet.text).sentiment
#csvFile.close()
public_tweets = api.search('CSR')

for tweet in public_tweets:
    print(tweet.text)
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment, tweet.created_at)
    csvWriter.writerow([tweet.created_at,tweet.text.encode('utf-8'),TextBlob(tweet.text).sentiment])
    print("")
csvFile.close()
