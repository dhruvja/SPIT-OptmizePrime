import streamlit as st
from datetime import date, timedelta

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
from nsetools import Nse

import bs4 
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup 

import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from streamlit_echarts import st_echarts
import time

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from threading import *
import pandas as pd

def app():


	START = "2015-01-01"
	END = date.today().strftime("%Y-%m-%d")

	st.title("Stock Prediction App")

	stocks = ("^BSESN", "^NSEI", "SBIN.NS", "TATAELXSI.NS", "RELIANCE.NS", "INFY.NS", "YESBANK.NS", "SUZLON.NS", "ZOMATO.NS")

	FullNames = {"^BSESN": "Sensex", "^NSEI": "Nifty", "SBIN.NS": "SBI India", "TATAELXSI.NS": "TATA Elxsi", "RELIANCE.NS": "Reliance", "INFY.NS": "Infosys", "YESBANK.NS": "Yes Bank", "SUZLON.NS": "Suzlon", "ZOMATO.NS": "Zomato"}

	selected_stocks = st.sidebar.selectbox(
		"Select The Stock Ticker",
		stocks

	)

	n_years = st.sidebar.slider("Years of prediction", 1,10)
	period = n_years*365

	statistics = st.empty()

	newsPositive = 0
	tweetsPositive = 0
	newsNegative = 0
	tweetsNegative = 0



	@st.cache(allow_output_mutation=True)
	def load_data(ticker):
		data = yf.download(ticker,START, END, interval="1d")
		data.reset_index(inplace=True)
		return data

	data = load_data(selected_stocks)
	placeholder = st.empty()
	predict = st.empty()

	class TwitterClient(object):
		def __init__(self):

			consumer_key = 'C8RUGu6BIj0ivXa4RFW6a2XP1'
			consumer_secret = 'JEC41aZwy0GY7OIrpa18CVWF8sPn2W0FvInM0xlg9may79E8hV'
			access_token = '881871842293698560-HVRzMgIbLvKWMHe6epnpTOTE5jv2zaU'
			access_token_secret = 'lnobRVJigV0EYOKYrj6a28goZr5SyIz9KMPVC4DPPFDPt'

			try:
				self.auth = OAuthHandler(consumer_key, consumer_secret)
				self.auth.set_access_token(access_token, access_token_secret)
				self.api = tweepy.API(self.auth)
			except:
				print("Error: Authentication Failed")

		def clean_tweet(self, tweet):
			return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

		def get_tweet_sentiment(self, tweet):
			analysis = TextBlob(self.clean_tweet(tweet))
			if analysis.sentiment.polarity > 0:
				return 'positive'
			elif analysis.sentiment.polarity == 0:
				return 'neutral'
			else:
				return 'negative'

		def get_tweets(self, query, count = 5):
			tweets = []

			try:
				fetched_tweets = self.api.search_tweets(q = query, count = count)

				for tweet in fetched_tweets:
					parsed_tweet = {}

					parsed_tweet['text'] = tweet.text
					parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

					if tweet.retweet_count > 0:
						if parsed_tweet not in tweets:
							tweets.append(parsed_tweet)
					else:
						tweets.append(parsed_tweet)

				return tweets

			except tweepy.TwitterServerError as e:
				print("Error : " + str(e))

	class Tweet(Thread):
		def run(self):
			print("Twitter analysis happening")

			api = TwitterClient()
			tweets = api.get_tweets(query = FullNames[selected_stocks], count = 20000)

			ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
			positive = 100*len(ptweets)/len(tweets)
			ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
			negative = 100*len(ntweets)/len(tweets)
			neutral = 100*(len(tweets) -(len( ntweets )+len( ptweets)))/len(tweets)

			st.title("Tweets Analysis")

			options = {
			"xAxis": {
				"type": "category",
				"data": ["Bullish", "Bearish", "Neutral"],
			},
			"yAxis": {"type": "value"},
			"series": [{"data": [{"value": positive, "itemStyle": {"color": 'green'}},{"value": negative, "itemStyle": {"color": 'red'}},{"value": neutral, "itemStyle": {"color": 'blue'}}], "type": "bar"}],
			}
			st_echarts(options=options, height="500px")  

			tweetsPositive = positive
			tweetsNegative = negative

			with st.expander("Bullish Tweets"):
				for tweet in ptweets[:10]:
					st.subheader(tweet['text'])
			
			with st.expander("Bearish Tweets"):
				for tweet in ntweets[:10]:
					st.subheader(tweet['text'])
			return (tweetsPositive, tweetsNegative)

	class RawData():
		def run(self):
			print("Raw data happening")
			print(type(data['Close']))
			print(data['Close'][data['Close'].size - 1])
			s1 = pd.Series([1908.87])
			s4 = data['Close'].append(s1, ignore_index=True)
			print(s4)
			# while True:

			with placeholder.container():
				st.subheader("Raw Data")
				s1 = pd.Series([np.random.choice(range(1,5))])
				s4 = data['Close'].append(s1, ignore_index=True)
				s5 = data['Open'].append(s1, ignore_index=True)
				# data['Date'] = data['Open'].append(s1, ignore_index=True)
				# print(data['Close'])
				st.write(data.tail())
				# data['Close'] = data['Close']*np.random.choice(range(1,5))
				# data['Open'].append(3*np.random.choice(range(1,5)))
				selected_stock = "BTC-USD"
				r = requests.get('https://finance.yahoo.com/quote/' + selected_stock + '?p=' + selected_stock)
				soup = bs4.BeautifulSoup(r.text,'lxml')
				price = soup.find('div',{'class':'My(6px) Pos(r) smartphone_Mt(6px) W(100%) D(ib) smartphone_Mb(10px) W(100%)--mobp'}).find('span').text
				st.subheader("Current Price")
				st.subheader(data['Close'][data['Close'].size-1])
				def plot_raw_data():
					fig = go.Figure()
					fig.add_trace(go.Scatter(x=data['Date'], y = s5, name="stock_open"))
					fig.add_trace(go.Scatter(x=data['Date'], y = s4, name="stock_close"))
					fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
					st.plotly_chart(fig)
				plot_raw_data()
				time.sleep(1)

	class Predict():
		def run(self):
			print("Prediction happening")
			df_train = data[['Date', 'Close']]
			df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

			m = Prophet()
			m.fit(df_train)
			future = m.make_future_dataframe(periods = period)
			forecast = m.predict(future)

			st.subheader("Forecast Data")
			st.write(forecast.tail())

			st.write('Forecast data')
			fig1 = plot_plotly(m, forecast)
			st.plotly_chart(fig1)

			st.write('Forecast data components')
			fig2 = m.plot_components(forecast)
			st.write(fig2)

			return forecast


	class NewsAnalyzer(Thread):
		def run(self):
			print("News analysis happening")
			nltk.download('vader_lexicon') #required for Sentiment Analysis
			now = date.today()
			now = now.strftime('%m-%d-%Y')
			yesterday = date.today() - timedelta(days = 1)
			yesterday = yesterday.strftime('%m-%d-%Y')
			nltk.download('punkt')
			user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15'
			config = Config()
			config.browser_user_agent = user_agent
			config.request_timeout = 5

			# save the company name in a variable
			company_name = selected_stocks
			company_name = FullNames[selected_stocks]
			#As long as the company name is valid, not empty...
			if company_name != '':
				#print(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')
				#Extract News with Google News
				googlenews = GoogleNews(start=yesterday,end=now)
				googlenews.search(company_name)
				result = googlenews.result()
				#store the results
				df = pd.DataFrame(result)
				#print(df)
			try:
				list =[] #creating an empty list 
				for i in df.index:
					dict = {} #creating an empty dictionary to append an article in every single iteration
					article = Article(df['link'][i],config=config) #providing the link
					try:
						article.download() #downloading the article 
						article.parse() #parsing the article
						article.nlp() #performing natural language processing (nlp)
					except:
						pass 
					#storing results in our empty dictionary
					dict['Date']=df['date'][i] 
					dict['Media']=df['media'][i]
					dict['Title']=article.title
					dict['Article']=article.text
					dict['Summary']=article.summary
					dict['Key_words']=article.keywords
					list.append(dict)
				check_empty = not any(list)
				# print(check_empty)
				if check_empty == False:
					news_df=pd.DataFrame(list) #creating dataframe
				#print(news_df)
			except Exception as e:
					#exception handling
					print("exception occurred:" + str(e))
					print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.' )
				#Sentiment Analysis
			def percentage(part,whole):
				return 100 * float(part)/float(whole)

			#Assigning Initial Values
			positive = 0
			negative = 0
			neutral = 0
			#Creating empty lists
			news_list = []
			neutral_list = []
			negative_list = []
			positive_list = []

			#Iterating over the tweets in the dataframe
			for news in news_df['Summary']:
				news_list.append(news)
				analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
				neg = analyzer['neg']
				neu = analyzer['neu']
				pos = analyzer['pos']

				if neg > pos:
					negative_list.append(news) #appending the news that satisfies this condition
					negative += 1 #increasing the count by 1
				elif pos > neg:
					positive_list.append(news) #appending the news that satisfies this condition
					positive += 1 #increasing the count by 1
				elif pos == neg:
					neutral_list.append(news) #appending the news that satisfies this condition
					neutral += 1 #increasing the count by 1 

			positive = percentage(positive, len(news_df)) #percentage is the function defined above
			negative = percentage(negative, len(news_df))
			neutral = percentage(neutral, len(news_df))

			#Converting lists to pandas dataframe
			news_list = pd.DataFrame(news_list)
			neutral_list = pd.DataFrame(neutral_list)
			negative_list = pd.DataFrame(negative_list)
			positive_list = pd.DataFrame(positive_list)
			#using len(length) function for counting
			print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
			print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
			print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')

			postive = (len(positive_list)/len(news_list))*100
			negative = (len(negative_list)/len(news_list))*100
			neutral = (len(neutral_list)/len(news_list))*100

			newsNegative = negative
			newsPositive = positive

			st.title("News Analysis")

			options = {
			"xAxis": {
				"type": "category",
				"data": ["Bullish", "Bearish", "Neutral"],
			},
			"yAxis": {"type": "value"},
			"series": [{"data": [{"value": positive, "itemStyle": {"color": 'green'}},{"value": negative, "itemStyle": {"color": 'red'}},{"value": neutral, "itemStyle": {"color": 'blue'}}], "type": "bar"}],
			}


			st_echarts(options=options, height="500px") 

			print(positive_list[0][0])

			try:
				with st.expander("Bullish News"):
					for news in positive_list[0]:
						st.subheader(news)
				
				with st.expander("Bearish News"):
					for news in negative_list[0]:
						st.subheader(news)
			except:
				print("No news")
			return (newsNegative, newsPositive)

	class Statistics():
		def run(self):
			weekly = yf.download(selected_stocks,START, END, interval="1wk")
			weekly.reset_index(inplace=True)
			# print(weekly['Close'][weekly['Close'].size - 2])
			# weekly['delta'] = delta = weekly['Close'].diff()
			# weekly['up'] = up = delta.clip(lower=0)
			# weekly['down'] = down = delta.clip(upper=0)

			# ema_up = up.ewm(com=13,adjust = False).mean()
			# ema_down = down.ewm(com=13,adjust = False).mean()

			# rs = ema_up/ema_down
			# weekly['RSI'] = 100 - (100/(1+rs))
			# print(weekly['RSI'])

			# def plot_raw_data():
			# 	fig = go.Figure()
			# 	fig.add_trace(go.Scatter(x=weekly['Date'], y = weekly['RSI'], name="RSI index"))
			# 	fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
			# 	st.plotly_chart(fig)
			# plot_raw_data()


			print(forecastedData['trend'])	
			ratio = forecastedData['trend'][forecastedData['trend'].size - 1]/data['Close'][data['Close'].size - 1]
			print(ratio)
			print(data['Close'].size)
			print(newsPositive, tweetsPositive)

			pos = (newsPositive + tweetsPositive)/2
			neg = (newsNegative + tweetsNegative)/2

			print((pos-neg)*ratio)

			ratio = (pos-neg)/ratio
			ratio = "{:.2f}".format(ratio)


			if weekly['Close'][weekly['Close'].size - 2] < weekly['Open'][weekly['Open'].size - 2]:
				with statistics.container():

					kpi1,kpi2 = st.columns(2)
					kpi1.metric(label="Circuit", value="Lower Circuit")
					kpi2.metric(label="Buy Percentage", value = str(ratio) + "%")
					# kpi2.metric(label="Married Count 💍", value= 8)
					# kpi3.metric(label="A/C Balance ＄", value= 7)
			else:
				with statistics.container():
					kpi1,kpi2= st.columns(2)
					kpi1.metric(label="Circuit", value="Upper Circuit")
					kpi2.metric(label="Buy Percentage", value = str(ratio) + "%")
					# kpi2.metric(label="Married Count 💍", value= 8)
					# kpi3.metric(label="A/C Balance ＄", value= 7)

	t1 = Predict()
	t2 = RawData()
	t3 = Tweet()
	t4 = NewsAnalyzer()
	t5 = Statistics()

	# t2.start()
	# t3.start()
	# t1.start()
	# print("waiting for it to return")
	# t4.start()
	tweetsPositive, tweetsNegative =  t3.run()
	t2.run()
	newsNegative, newsPositive = t4.run()
	forecastedData = t1.run()
	t5.run()

