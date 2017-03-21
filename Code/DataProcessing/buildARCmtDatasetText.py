import numpy as np
import re
from logProgress import *

'''
	Description : The file builds the ARCmtDataset_text using the ARCmtDataset.
	Input		: ARCmtDataset.txt (Dataset in comment_name format)
	Ouput       : ARCmtDataset_text.txt (Dataset in comment_body format)

'''

data_path = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/"
file_names = ['NewPrunedDataset/ARCmtDataset_test.txt','NewPrunedDataset/ARCmtDataset_val.txt','NewPrunedDataset/ARCmtDataset_train.txt']

## Pruning rules
# 1. URL present in comment or context
# 2. Comment with length >= 50
# 3. Comment with length <= 3
# 4. 'reddit' present in body

def IgnoreBody(body):
	body = body.lower()
	if "http" in body or "www" in body or "reddit" in body:
		return True
	return False

## Make body dict
print 'Starting with Body Dict Building ...'

body_dict = {}
all_comments = open(data_path + 'UsefulComments_NoChildren.txt').readlines()
for i in logProgress(xrange(len(all_comments))):
	comment_i = eval(all_comments[i])
	body_string = comment_i['body'].translate(None, '\t\n\r').strip()
	all_comments[i] = ''
	if IgnoreBody(body_string) == False:
		body_dict[comment_i['name']] = body_string

## Prune and write to file
print 'Starting with ARTCmtDataset Now ...'

DELIMITER = '<CTX>'
for input_file_name in file_names:
	
	## Read Data in comment_name format
	all_data_points = open(data_path + input_file_name).readlines()
	for i in logProgress(xrange(len(all_data_points))):
		all_data_points[i] = all_data_points[i].split('\t')
		all_data_points[i][0] = eval(all_data_points[i][0])

	## Dump data to file
	with open(data_path + input_file_name,'w') as f:
		for comment in logProgress(all_data_points):
			buf = ""
			for context_id in comment[0][::-1]:
				if context_id in body_dict:
					buf += DELIMITER + ' ' + body_dict[context_id]
			if buf == "":
				continue
			if comment[1] in body_dict and comment[2] in body_dict:
				len_resp_1 = len(body_dict[comment[1]].split(' '))
				len_resp_2 = len(body_dict[comment[1]].split(' '))
				if len_resp_1 >= 50 or len_resp_2 >= 50:
					continue
				if len_resp_1 <= 3 or len_resp_2 <= 3:
					continue
				# To create only Pos/Neg dataset
				# if comment[3] == '0\n':
				# if comment[3] == '1\n':
				# 	continue
				buf += '\t' + body_dict[comment[1]] + '\t' + body_dict[comment[2]] + '\t' + comment[3]
				if len(buf.split('\t')) != 4:
					continue
				f.write(buf)
