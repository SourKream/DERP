import numpy as np
import cPickle as pickle
import random
from logProgress import *

'''
	Description : The file builds the ARCmtDataset using the ChildrenDict. The output 
				  from this file is in comment_name format. To convert the output to 
				  text_format, use buildARCmtDatasetText.py
	Ouput       : ARCmtDataset.txt (Dataset in comment_name format)

'''

data_path = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/"

## Functions
def getPermutationPairs(num):
	pairs = []
	for i in range(num):
		for j in range(i+1, num):
			pairs.append((i, j))
	random.shuffle(pairs)
	return pairs[:6]

## Load Data
print "Loading data..."
child_dict = pickle.load(open(data_path+'ChildrenDict.pkl'))
parent_dict = pickle.load(open(data_path+'Name2ParentId.pkl'))
link_id = pickle.load(open(data_path+'Name2LinkId.pkl'))

## Permutations of child pairs to use
selected_pairs = {}
for i in range(2, 11):
	selected_pairs[i] = getPermutationPairs(i)

## Random comments list for bad child
random_comments = []
for comment_name in child_dict:
	random_comments.extend(child_dict[comment_name])
random.shuffle(random_comments)

## Extract good and bad pairs
useful_comments = []
for comment_name in logProgress(child_dict):
	num_child_comments = len(child_dict[comment_name])
	if num_child_comments >= 2 and num_child_comments <= 10:
		curr_dict = {}

		## Find true child comment pairs
		curr_dict['true_child_pairs'] = []
		for pair in selected_pairs[num_child_comments]:
			gold_child = child_dict[comment_name][pair[0]]
			alternate_child = child_dict[comment_name][pair[1]]
			curr_dict['true_child_pairs'].append((gold_child, alternate_child))

		## Find false comment pairs
		curr_dict['false_child_pairs'] = []
		for pair in selected_pairs[num_child_comments]:
			gold_child = child_dict[comment_name][pair[0]]
			bad_child = random_comments.pop()
			curr_dict['false_child_pairs'].append((gold_child, bad_child))

		## Find entire context
		curr_dict['context'] = []
		curr = comment_name
		while curr in parent_dict and parent_dict[curr] != link_id[curr]:
			curr_dict['context'].append(curr)
			curr = parent_dict[curr]
		curr_dict['context'].append(curr)
		curr_dict['context'].append(parent_dict[curr])

		useful_comments.append(curr_dict)

## Dump to file
with open(data_path+'ARCmtDataset.txt','w') as f:
	for comment in useful_comments:
		for pair in comment['true_child_pairs']:
			f.write(str(comment['context'])+'\t')
			f.write(pair[0]+'\t')
			f.write(pair[1]+'\t')
			f.write('1\n')
		for pair in comment['false_child_pairs']:
			f.write(str(comment['context'])+'\t')
			f.write(pair[0]+'\t')
			f.write(pair[1]+'\t')
			f.write('0\n')
