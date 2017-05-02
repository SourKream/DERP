import tweepy
import sys
import re
from logProgress import *

auth = tweepy.OAuthHandler('vvzSvqj1CfWJ3c63qjzC0fFS9', 'QBl6GFtcaco8AaRoPbRKcoEgSYSXbz1HFrwOwuBDDJVojH3Xdw')
auth.set_access_token('842402969312096257-q33j8tm5VJGP6Z2jA4HusPsOuwwvB4c', 'Sw74OHKh2QXWz0cpfMUmUaqYMfvQZk94yVzmu95GdcK9s')
api = tweepy.API(auth)

input_file = '/Users/Akshay/Downloads/TwitterDialogueCorpus/TweetIDs/TweetIDs_Test.txt'
output_file = '/Users/Akshay/Downloads/TwitterDialogueCorpus/TweetIDs/Tweets_Test.txt'
if len(sys.argv) > 1:
	input_file = sys.argv[1]
if len(sys.argv) > 2:
	output_file = sys.argv[2]

f = open(input_file).readlines()

# batch tweet id's into lists of up to 100 for batch-querying
tweet_id_lists = []
conv_id_lists = []
tweet_id_list = list()
tweet_id_statuses_list = []
for line in f:
	conv_id_list = line.strip().split('\t')
	if len(conv_id_list) > 100:
		conv_id_list = conv_id_list[-100:]
	conv_id_lists.append(conv_id_list)
	if len(tweet_id_list) + len(conv_id_list) > 100:
		tweet_id_lists.append(tweet_id_list)
		tweet_id_list = list()
	tweet_id_list.extend(conv_id_list)

# get all the tweets
for id_list in logProgress(tweet_id_lists):
	tweet_id_statuses = api.statuses_lookup(id_list)
	tweet_id_statuses_list.extend(tweet_id_statuses)

# batch querying returns out of order tweets, need to reorder according to original query id's
tweet_id_statuses_list_new = []
begin = 0
for i, id_list in enumerate(tweet_id_lists):
	matched = 0
	end = len(id_list) + begin
	for tweet_id in id_list:
		found = False
		for status_id in tweet_id_statuses_list[begin:end]:
			if status_id.id_str == tweet_id:
				tweet_id_statuses_list_new.append(status_id)
				found = True
				break;
		if found:
			matched += 1
	begin += matched
tweet_id_statuses_list = tweet_id_statuses_list_new

# match tweet id's and build tweet conversations
k = 0
conv_lists = []
for i, conv_id_list in enumerate(conv_id_lists):
	conv_exists = True
	conv_lists.append([])
	for j, tweet_id in enumerate(conv_id_list):
		if k < len(tweet_id_statuses_list) and tweet_id == tweet_id_statuses_list[k].id_str:
			if conv_exists:
				text = tweet_id_statuses_list[k].text.encode('utf-8')
				text = ' '.join(re.split('[\n\t]', text.strip()))
				conv_lists[-1].append(text)
			k += 1
		elif conv_exists:
			conv_lists.pop()
			conv_lists.append(['MISSING CONVERSATION'])
			conv_exists = False

g = open(output_file, 'w')

# write conversations to file
for conv_list in conv_lists:
	g.write('\t'.join(conv_list))
	g.write('\n')

g.close()