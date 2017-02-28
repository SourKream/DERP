import numpy as np
import cPickle as pickle

## Logic for generating Dataset
# 1. Choose comments with 2 or more children comments
# 2. Clean/Filter 
# 3. Find entire history till original post (API needed?)
# 4. Choose wrong response (random?)

data_path = '~/scratch/DERP/Reddit/'

child_dict = pickle.load(open(data_path+'ChildrenDict.pkl'))
parent_dict = pickle.load() ## Create Parent Dict
link_id = pickle.load() ## Create Link_id Dict
body = pickle.load() ## Create Body Dict

useful_comments = []

for comment_name in child_dict:
	num_child_comments = len(child_dict[comment_name])
	if num_child_comments >= 2 and num_child_comments <= 10:
		curr_dict = {}

		## Find child comment pairs
		curr_dict['true_child_pairs'] = []
		first_child = child_dict[comment_name][0]
		for other_child in child_dict[comment_name][1:]:
			curr_dict['true_child_pairs'].append((first_child, other_child))

		## Find false comment pairs

		## Find entire Contect
		curr_dict['context'] = []
		curr = comment_name
		while parent_dict[curr] != link_id[curr]:
			curr_dict['context'].append(curr)
			curr = parent_dict[curr]
		curr_dict['context'].append(curr)
		curr_dict['context'].append(parent_dict[curr])

		useful_comments.append(curr_dict)

## Get body of context and comments

## Dump to file
with open(data_path+'ARCmtDataset.txt','w') as f:
	for comment in useful_comments:
		for pair in comment['true_child_pairs']:
			f.write(str(comment['context']))
			f.write(pair[0]+'\t')
			f.write(pair[1]+'\t')
			f.write('1\n')


