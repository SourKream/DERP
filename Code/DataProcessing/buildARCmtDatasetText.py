import numpy as np

'''
	Description : The file builds the ARCmtDataset_text using the ARCmtDataset.
	Input		: ARCmtDataset.txt (Dataset in comment_name format)
	Ouput       : ARCmtDataset_text.txt (Dataset in comment_body format)

'''

data_path = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/"

## Read Data in comment_name format
print 'Starting with ARTCmtDataset Now ...'

idx_data = []
uid_set = set([])
counter = 0
for line in open(data_path + "ARCmtDataset.txt"):
	line = line.split('\t')
	line[0] = eval(line[0])
	uid_set.update(line[0])
	uid_set.add(line[1])
	uid_set.add(line[2])
	idx_data.append(line)
	counter += 1
	if counter % 1000000 == 0:
		print counter 

## Make body dict
print 'Starting with Body Dict Building ...'

def filterByTab(text):
	return ''.join(filter(lambda x: x!= '\t' and x !='\n', list(text)))

body_dict = {}
counter = 0
for line in open(data_path + 'UsefulComments_NoChildren.txt'):
	line = eval(line)
	if line['name']	in uid_set:
		body_dict[line['name']] = filterByTab(line['body'])
		counter += 1
		if counter % 1000000 == 0:
			print counter

## Dump data to file
print 'Starting with writing to file ...'

DELIMITER = '<CTX>'
counter = 0
with open(data_path+'ARCmtDataset_text.txt','w') as f:
	for comment in idx_data:
		buf = ""
		for context_id in comment[0]:
			if context_id in body_dict:
				buf += ' ' + DELIMITER + ' ' + body_dict[context_id]
		if buf == "": 
			continue
		if comment[1] in body_dict and comment[2] in body_dict:
			buf += '\t' + body_dict[comment[1]] + '\t' + body_dict[comment[2]] + '\t' + comment[3]
			f.write(buf)
			counter += 1
			if counter % 1000000 == 0:
				print counter

