import numpy as np
import re
from logProgress import *

'''
	Description : The file builds the ARCmtDataset_text using the ARCmtDataset.
	Input		: ARCmtDataset.txt (Dataset in comment_name format)
	Ouput       : ARCmtDataset_text.txt (Dataset in comment_body format)

'''

data_path = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/"
input_file_name = 'NewPrunedDataset/ARCmtDataset.txt'
output_file_name = 'NewPrunedDataset/ARCmtDataset_text_Pruned.txt'

## Read Data in comment_name format
print 'Starting with ARTCmtDataset Now ...'

idx_data = []
uid_set = set([])
for line in logProgress(open(data_path + input_file_name), every=1000):
	line = line.split('\t')
	line[0] = eval(line[0])
	uid_set.update(line[0])
	uid_set.add(line[1])
	uid_set.add(line[2])
	idx_data.append(line)

## Make body dict
print 'Starting with Body Dict Building ...'

def filterByTab(text):
	return ''.join(filter(lambda x: x!= '\t' and x !='\n', list(text))).strip()

body_dict = {}
for line in logProgress(open(data_path + 'UsefulComments_NoChildren.txt'), every=1000):
	line = eval(line)
	if line['name']	in uid_set:
		body_string = filterByTab(line['body'])
		if body_string != '':
			body_dict[line['name']] = body_string

## Pruning rules
# 1. URL present in comment or context
# 2. Comment with length >= 40
# 3. Comment with length <= 3

def URLPresent(body):
	if "http" in body or "www" in body:
		return True
	return False

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
with open(data_path+output_file_name,'w') as f:
	for comment in logProgress(idx_data):
		buf = ""
		for context_id in comment[0]:
			if context_id in body_dict:
				if URLPresent(body_dict[context_id]):
					continue
				buf += DELIMITER + ' ' + body_dict[context_id]
		if buf == "": 
			continue
		if comment[1] in body_dict and comment[2] in body_dict:
			if URLPresent(body_dict[comment[1]]) or URLPresent(body_dict[comment[2]]):
				continue

			len_resp_1 = len(tokenize(body_dict[comment[1]]))
			len_resp_2 = len(tokenize(body_dict[comment[1]]))
			if len_resp_1 >= 50 or len_resp_2 >= 40:
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

