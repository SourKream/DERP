import progressbar
import sys
from random import shuffle
if __name__ == "__main__":
	DATA_FILE = "/scratch/cse/btech/cs1130773/DERPData/Data/train_with_incorrect_grammar.txt"
	ctx_resp = {}
	for line in open(DATA_FILE):
		line = line.strip().split('\t')
		if line[0] not in ctx_resp:
			ctx_resp[line[0]] = [set(),set()]
		ctx_resp[line[0]][1].add(line[1])
		ctx_resp[line[0]][int(line[-1])].add(line[2])
	print 'CTX RESP building done ... '
	sys.stdout.flush()
	data = []
	bar = progressbar.ProgressBar(max_value = len(ctx_resp))
	for idx,ctx in enumerate(ctx_resp):
		for pos in ctx_resp[ctx][1]:
			for neg in ctx_resp[ctx][0]:
				if pos.strip() != '' and neg.strip() != '':
					data.append(ctx + '\t' + pos + '\t' + neg)
		bar.update(idx)
	shuffle(data)
	for elem in data:
		assert len(elem.split('\t')) == 3
	
	OUT_FILE = "/scratch/cse/btech/cs1130773/DERPData/Data/RUBER_with_incorrect_grammar_train.txt"
	open(OUT_FILE,'wb').write('\n'.join(data))
	

