import progressbar
from random import shuffle
import sys
if __name__=="__main__":
	DATA_FILE = "/scratch/cse/btech/cs1130773/DERPData/Data/train_with_incorrect_grammar.txt"
	print 'LOADING DATA FROM : ',DATA_FILE	
	sys.stdout.flush()
	THRESHOLD = 15000000
	ctx_gold_resp = {}
	for line in open(DATA_FILE):
		line = line.strip().split('\t')
		if line[0] not in ctx_gold_resp:
			ctx_gold_resp[line[0]] = [set(), set(), set()]
		ctx_gold_resp[line[0]][2].add(line[1])
		ctx_gold_resp[line[0]][int(line[-1])].add(line[2])
	print 'CTX RESP generation done ... '
	bar = progressbar.ProgressBar(max_value = len(ctx_gold_resp))
	data = []
	for idx,ctx in enumerate(ctx_gold_resp):
		cur_ctx = ctx_gold_resp[ctx]
		for gold in cur_ctx[2]:
			prefix = ctx + '\t' + gold
			for pos in cur_ctx[1]:
				for neg in cur_ctx[0]:
					if pos.strip() != '' and neg.strip() != '':
						data.append(prefix + '\t' + pos + '\t' + neg)
		bar.update(idx)
	shuffle(data)	
	print 'SHUFFLING OF DATA DONE ...'
	sys.stdout.flush()
	data = data[:THRESHOLD]	
	OUT_FILE = "/scratch/cse/btech/cs1130773/DERPData/Data/RUBER_GOLD_train_with_incorrect_grammar.txt"
	f = open(OUT_FILE, 'wb')
	bar = progressbar.ProgressBar(max_value = len(data))
	for idx,elem in enumerate(data):
		assert len(elem.split('\t')) == 4
		if idx == 0:
			f.write(elem)
		else:
			f.write ("\n" + elem)
		bar.update(idx)
	# open(OUT_FILE, 'wb').write('\n'.join(data))
