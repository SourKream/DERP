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

## Read Data in comment_name format
print 'Starting with ARTCmtDataset Now ...'

idx_data = {}
for input_file_name in file_names:
	all_data_points = open(data_path + input_file_name).readlines()
	for i in logProgress(xrange(len(all_data_points))):
		all_data_points[i] = all_data_points[i].split('\t')
		all_data_points[i][0] = eval(all_data_points[i][0])
	idx_data[input_file_name] = all_data_points[i]

## Make body dict
print 'Starting with Body Dict Building ...'

def IgnoreBody(body):
	body = body.lower()
	if "http" in body or "www" in body or "reddit" in body:
		return True
	return False

body_dict = {}
all_comments = open(data_path + 'UsefulComments_NoChildren.txt').readlines()
for i in logProgress(xrange(len(all_comments))):
	all_comments[i] = eval(all_comments[i])
	body_string = all_comments[i]['body'].translate(None, '\t\n\r').strip()
	if body_string == '' or IgnoreBody(body_string):
		continue
	body_dict[all_comments[i]['name']] = body_string
del all_comments

## Pruning rules
# 1. URL present in comment or context
# 2. Comment with length >= 50
# 3. Comment with length <= 3
# 4. 'reddit' present in body

def tokenize(sent):
    ## Add exception regex enclosed in ()
    exceptions = ['(corp\.)', '(ave\.)', '(a\.k\.a)', '(mr\.)', '(jr\.)', 
    			  '(gen\.)', '(feb\.)', '(a\.k\.a\.)', '(km\.)', '(mrs\.)', 
    			  '(dc\.)', '(p\.s)', '(pp\.)', '(j\.r\.)', '(b\.s\.)', 
    			  '(etc\.)', '(vol\.)', '(alex\.)', '(ex\.)', '(d\.c\.)', 
    			  '(b\.c\.)', '(bros\.)', '(op\.)', '(co\.)', '(r\.i\.p)', 
    			  '(j\.j\.)', '(sp\.)', '(mins\.)', '(\.s)', '(\.\.\.)', 
    			  '(dept\.)', '(vs\.)', '(kg\.)', '(a\.i\.)', '(gov\.)', 
    			  '(states\.)', '(j\.)', '(sec\.)', '(k\.)', '(\.k\.)', 
    			  '(c\.k\.)', '(ms\.)', '(p\.m\.)', '(reps\.)', '(nov\.)', 
    			  '(iv\.)', '(hrs\.)', '(ca\.)', '(sgt\.)', '(mon\.)', 
    			  '(jan\.)', '(sr\.)', '(oz\.)', '(corps\.)', '(bc\.)', 
    			  '(dr\.)', '(min\.)', '(prof\.)', '(ga\.)', '(pts\.)', 
    			  '(lb\.)', '(rev\.)', '(mi\.)', '(ph\.d\.)', '(ad\.)', 
    			  '(p\.s\.)', '(l\.a\.)', '(ft\.)', '(amazon\.com)', 
    			  '(e\.g)', '(ed\.)', '(sc\.)', '(lt\.)', '(i\.e\.)', 
    			  '(inc\.)', '(va\.)', '(r\.i\.p\.)', '(est\.)', '(la\.)', 
    			  '(govt\.)', '(mt\.)', '(\.s\.)', '(i\.e)', '(st\.)', 
    			  '(google\.com)', '(t\.v\.)', '(a\.m\.)', '(cal\.)', 
    			  '(sat\.)', '(dec\.)', '(cons\.)', '(rep\.)', '(lbs\.)', 
    			  '(mo\.)', '(e\.g\.)', '(j\.k\.)']
    return [''.join(x) for x in re.findall("|".join(exceptions)+"|([0-9]+)|([\w']+)|([.,!?;])",sent)]

## Dump data to file
print 'Starting with writing to file ...'

DELIMITER = '<CTX>'

for output_file_name in file_names:
	with open(data_path+output_file_name,'w') as f:
		for comment in logProgress(idx_data[output_file_name]):
			buf = ""
			for context_id in comment[0][::-1]:
				if context_id in body_dict:
					buf += DELIMITER + ' ' + body_dict[context_id]
			if buf == "":
				continue
			if comment[1] in body_dict and comment[2] in body_dict:
				len_resp_1 = len(tokenize(body_dict[comment[1]]))
				len_resp_2 = len(tokenize(body_dict[comment[1]]))
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

