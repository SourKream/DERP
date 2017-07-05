from scipy import spatial
from sklearn.metrics import confusion_matrix,accuracy_score
import sys
import pdb
## Some Initializations
sys.path.append('/home/cse/dual/cs5130275/DERP/Code')
import SkipThoughts.skipthoughts as skt

def getParams():
	options = {}
	options['BATCH_SIZE'] = 500
	options['THRESHOLD'] = 0.5
	#options['Model'] = None 	
	options['Model'] = skt.load_model()
	options['Local'] = False
	options['DEBUG'] = 0
	#options['METRIC'] = 'cosine' # TODO : Make more generic
	
	return options
def process_batch(predicted_t, gold_t, predicted_labels, opts):
	#opts['DEBUG'] += 1
	#print 'Batch Counter: ', counter
	#print '#DEBUG2'
	predicted_array = skt.encode(opts['Model'], predicted_t, verbose = False, batch_size=opts['BATCH_SIZE'])
	#print '#DEBUG3'
	#if opts['DEBUG'] == 19:
		#print 'This gets weird'
		#print gold_t
		#open('weirdSentences.txt','wb').write('\n'.join(gold_t))
	gold_array = skt.encode(opts['Model'], gold_t, verbose = False, batch_size=opts['BATCH_SIZE'])
	#print '#DEBUG4'
	for idx in xrange(predicted_array.shape[0]):
		cosine_dist = spatial.distance.cosine(predicted_array[idx], gold_array[idx])
		if cosine_dist < opts['THRESHOLD']:
			predicted_labels.append(1)
		else:
			predicted_labels.append(0)

if __name__ == "__main__":
	opts = getParams()
	if opts['Local'] :
		data_dir = "../../"
	else:
		data_dir = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/DatasetWithPruning7M/"
	data_file = data_dir + "test_clean.txt"
	#data_file = data_dir  + "LogRegData/train_100K_clean.txt"
	gold_t = []
	predicted_t = []
	labels = []
	predicted_labels = []	
	print 'Using threshold', opts['THRESHOLD']
	sys.stdout.flush()
	print 'Starting with the data ...'
	sys.stdout.flush()
	with open(data_file) as f:
		for line in f:
			line = line.split('\t')
			gold = line[1]
			predicted = line[2]
			label = eval(line[3])
			labels.append(label)
			gold_t.append(gold)
			predicted_t.append(predicted)
			if len(gold_t) == opts['BATCH_SIZE']:
				#print '#DEBUG1'
				try:	
					process_batch(predicted_t, gold_t, predicted_labels, opts)
				except:
					pass
					#pdb.set_trace()
				if len(predicted_labels) != 0 and len(predicted_labels) % 10000 == 0:
					print len(predicted_labels), 'Done ...'
				sys.stdout.flush()
				gold_t = []
				predicted_t = []
		if len(gold_t) != 0:
			process_batch(predicted_t, gold_t, predicted_labels, opts)
			print len(predicted_labels), 'Done ...'
			gold_t = []
			predicted_t = []
	print 'Accuracy : ', accuracy_score(labels, predicted_labels)
	print 'Confusion Matrix'
	print confusion_matrix(labels, predicted_labels)
