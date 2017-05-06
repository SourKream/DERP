############
import progressbar
import time

def logProgress(sequence):

    try:
        bar = progressbar.ProgressBar(max_value=len(sequence))
    except TypeError:
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    for index, record in enumerate(sequence, 1):
        bar.update(index)
        yield record
#############

import gzip
import csv

parent_id = {}
body = {}

def IgnoreBody(body):
	body = body.lower()
	if "http" in body or "www" in body or "reddit" in body:
		return True
	if len(body.split(' ')) > 50:
		return True
	return False

numFiles = 72

for i in range(numFiles):
	print "File Num : ", i+1
	fil = gzip.open('TarFiles/2016-01-0000000000'+"%02d"%i+'.gz')
	fil_itr = csv.reader(fil)
	fil_itr.next()
	for comment in logProgress(fil_itr):
		cmt_id = 't1_' + comment[15]
		parent_id[cmt_id] = comment[10]

ctx = {}
for item in logProgress(parent_id):
	if parent_id[item] not in parent_id:
		continue
	if parent_id[parent_id[item]] not in parent_id:
		continue
	ctx[item] = [parent_id[parent_id[item]], parent_id[item], item]
	# temp = parent_id[item]
	# while temp in parent_id:
	# 	temp = parent_id[temp]
	# 	ctx[item].append(temp)

## Save without body
with open('3TurnDialogue_Reddit_IDX.txt','w') as filout:
	for item in logProgress(ctx):
		for idd in ctx[item]:
			filout.write(idd + '\t')
		filout.write('\n')

useful_idx = set([])
for item in logProgress(ctx):
	useful_idx.update(ctx[item])

for i in range(numFiles):
	print "File Num : ", i+1
	fil = gzip.open('TarFiles/2016-01-0000000000'+"%02d"%i+'.gz')
	fil_itr = csv.reader(fil)
	fil_itr.next()
	for comment in logProgress(fil_itr):
		if 't1_'+comment[15] in useful_idx:
			if not IgnoreBody(comment[0]):
				cmt_id = 't1_' + comment[15]
				body[cmt_id] = comment[0]

with open('3TurnDialogue_Reddit.csv','w') as filout:
	wr = csv.writer(filout)
	for item in logProgress(ctx):
		if ctx[item][0] not in body:
			continue
		if ctx[item][1] not in body:
			continue
		if ctx[item][2] not in body:
			continue
		wr.writerow([body[cmt_idx] for cmt_idx in ctx[item]])


