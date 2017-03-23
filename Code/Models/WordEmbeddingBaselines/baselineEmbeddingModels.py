import numpy as np
from my_utils import my_tokenize
from scipy import spatial
import gensim
import pdb
import sys
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def getParams():
	options = {}
	options['EMBEDDING_METHODS'] = ['GREEDY_MATCHING', 'EMBEDDING_AVERAGE', 'VECTOR_EXTREMA']
	# options['EMBEDDING_METHODS'] = ['EMBEDDING_AVERAGE']
	options['USE_CONTEXT'] = False
	options['LOCAL'] = True	
	
	if options['LOCAL']:
		data_dir = '../../../'
	else:
		data_dir = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/DatasetNewPruned11M/"
	data_file = data_dir + 'test.txt'
	options['DATA_FILE'] = data_file

	options['GLOVE'] = False
	options['WRITE2FILE'] = True
	
	options['THRESHOLD'] = 0.5 # DEFAULT VALUE
	if options['LOCAL']:
		if options['GLOVE']:
			options['MODEL_DIR'] = '/Users/barunpatra/Documents/glove.840B.300d.txt'
		else:
			options['MODEL_DIR'] = '../../../gensim_model.model'
		options['WRITE_DIR'] = '../../../'
	else:		
		if options['GLOVE']:
			options['MODEL_DIR'] = '/home/cse/dual/cs5130275/scratch/DERP/glove.840B.300d.txt'
		else:
			options['MODEL_DIR'] = '/home/cse/dual/cs5130275/scratch/DERP/Models/Gensim_Models/gensim_model.model'
		options['WRITE_DIR'] = '/home/cse/dual/cs5130275/DERP/Code/Models/WordEmbeddingBaselines/'
	options['PR_CURVE'] = True			

	return options

class BaseLineModels(object):
	def __init__(self, options):
		self.options = options
		self.confusion_matrices = {}
		self.MAX_DIST = 3.0

		for method in options['EMBEDDING_METHODS']:
			self.confusion_matrices[method] = np.zeros((2,2))		
		print 'STARTING TO LOAD THE MODEL'
		sys.stdout.flush()
		self.setModel()
		print 'FINISHED LOADING MODEL'
		sys.stdout.flush()
		if options['PR_CURVE']:
			self.predicted_labels = {}
			self.gold_labels = []
			for method in options['EMBEDDING_METHODS']:
				self.predicted_labels[method] = []
	
	def setModel(self):
		if options['GLOVE']:
			self.model = {}
			with open(self.options['MODEL_DIR']) as f:
				for line in f:
					line = line.strip().split(' ')
					self.model[line[0]] = np.array([float(x) for x in line[1:]])
		else:
			self.model = gensim.models.Word2Vec.load(self.options['MODEL_DIR'])
	
	def generateConfusionMatrices(self):
		counter = 0
		if self.options['LOCAL']:
			PRINT_THRESHOLD = 10
		else:
			PRINT_THRESHOLD = 10000
		with open(self.options['DATA_FILE']) as f:
			for line in f:				
				line = line.strip().split('\t')				
				if options['USE_CONTEXT']:
					# TODO : Fill this out
					pass
				else:
					_label = eval(line[3])
					if self.options['PR_CURVE']:						
						self.gold_labels.append(_label)
					for method in self.options['EMBEDDING_METHODS']:						
						self.updateConfusionMatrix(method, my_tokenize(line[1]), my_tokenize(line[2]), _label)
					counter += 1
					if (counter % PRINT_THRESHOLD) == 0:
						print counter, 'DONE'
						sys.stdout.flush()
		if self.options['PR_CURVE']:
			self.generateProbabilities()
	def generateProbabilities(self):
		for method in self.options['EMBEDDING_METHODS']:
			max_dist = -2
			for elem in self.predicted_labels[method]:
				if elem != self.MAX_DIST:
					max_dist = max(max_dist, elem)
			for idx in xrange(len(self.predicted_labels[method])):
				elem = self.predicted_labels[method][idx]
				self.predicted_labels[method][idx] = 0. if elem == self.MAX_DIST else 1. - (elem/ max_dist)

	def getGreedy(self,seq1,seq2):
		count = 0
		G = self.MAX_DIST
		for word1 in seq1:
			if word1 in self.model:
				embed_word1 = self.model[word1]
				cosine_dist = self.MAX_DIST
				for word2 in seq2:
					if word2 in self.model:
						cosine_dist = min(cosine_dist, spatial.distance.cosine(embed_word1, self.model[word2]) )				
				if G is None:
					G = 0.0
				G += cosine_dist
				count += 1
		if count != 0:
			G /= count
		return G
	
	def getSentenceEmbedding(self, sentence, USE_AVERAGE = True):
		if USE_AVERAGE == True:
			average = None
			if self.options['GLOVE']:
				average = np.zeros((300,))
			else:
				average = np.zeros((200,))
			count = 0
			for word in sentence:
				if word in self.model:
					average += self.model[word]
					count += 1
			if count != 0:
				average /= count
			return average
		else:
			ret_matrix = None
			for word in sentence:
				if word in self.model:
					word_embedding = self.model[word]
					if ret_matrix is None:
						ret_matrix = word_embedding
					else:
						for idx in xrange(ret_matrix.size):
							ret_matrix[idx] = word_embedding[idx] if ( abs(word_embedding[idx]) > abs(ret_matrix[idx]) ) else ret_matrix[idx]
			if ret_matrix is None:
				return np.zeros((300,)) if self.options['GLOVE'] else np.zeros((200,))
			return ret_matrix
			

	def updateConfusionMatrix(self, method, gold, predicted, gold_label):		
		predicted_label = -1			
		prediction_distance = -2
		
		if method == 'GREEDY_MATCHING':		
			G_predicted_gold = self.getGreedy(predicted, gold)
			G_gold_predicted = self.getGreedy(gold, predicted)
			G_score = (G_gold_predicted + G_predicted_gold) / 2			
			
			prediction_distance = G_score
			
			if G_score < self.options['THRESHOLD']:
				predicted_label = 1
			else:
				predicted_label = 0			

		elif method == 'EMBEDDING_AVERAGE':	
			predicted_embedding = self.getSentenceEmbedding(predicted)
			gold_embedding = self.getSentenceEmbedding(gold)
			if np.count_nonzero(predicted_embedding) == 0 or np.count_nonzero(gold_embedding) == 0:
				prediction_distance = self.MAX_DIST
				predicted_label = 0
			else:
				cosine_dist = spatial.distance.cosine(predicted_embedding, gold_embedding)
				prediction_distance = cosine_dist
				if cosine_dist < self.options['THRESHOLD']:
					predicted_label = 1
				else:
					predicted_label = 0				

		elif method == 'VECTOR_EXTREMA':
			predicted_embedding = self.getSentenceEmbedding(predicted, False)
			gold_embedding = self.getSentenceEmbedding(gold, False)
			if np.count_nonzero(predicted_embedding) == 0 or np.count_nonzero(gold_embedding) == 0:
				prediction_distance = self.MAX_DIST
				predicted_label = 0
			else:
				cosine_dist = spatial.distance.cosine(predicted_embedding, gold_embedding)				
				prediction_distance = cosine_dist
				if cosine_dist < self.options['THRESHOLD']:
					predicted_label = 1
				else:
					predicted_label = 0	
		self.confusion_matrices[method][gold_label][predicted_label] += 1
		if options['PR_CURVE']:
			self.predicted_labels[method].append(prediction_distance) 

if __name__ == "__main__":
	options = getParams()	
	if len(sys.argv) > 1:
		options['THRESHOLD'] = eval(sys.argv[1])
	model = BaseLineModels(options)
	model.generateConfusionMatrices()

	if options['PR_CURVE']:
		# precision = {}
		# recall = {}
		
		lw = 2
		counter = 1
		for key in model.predicted_labels:
			precision, recall, _ = precision_recall_curve(model.gold_labels, model.predicted_labels[key])
			#plt.clf()
			counter += 1
			plt.plot(recall, precision, lw=lw, color='navy', label='Precision-Recall curve')
			plt.xlabel('Recall')
			plt.ylabel('Precision')
			plt.ylim([0.0, 1.05])
			plt.xlim([0.0, 1.0])			
			plt.title('MODEL : ' + key)
			file_name = options['WRITE_DIR'] + "Image__" + key + "__.png"
			plt.savefig(file_name)


	writeBuf = "THRESHOLD: \t" +  str(options['THRESHOLD']) + '\n\n'
	for key in model.confusion_matrices:		
		writeBuf += 'METHOD: \t' + key + '\n'		
		accuracy = (model.confusion_matrices[key][1][1] + model.confusion_matrices[key][0][0]) / model.confusion_matrices[key].sum() * 100.
		precision = (model.confusion_matrices[key][1][1]) / (model.confusion_matrices[key][1][1] + model.confusion_matrices[key][0][1] ) * 100.
		recall = (model.confusion_matrices[key][1][1]) / (model.confusion_matrices[key][1][1] + model.confusion_matrices[key][1][0] ) * 100.
		fscore = (2*precision*recall)/ (precision + recall)
		writeBuf += 'ACCURACY : ' + '\t' +  str(accuracy) + "\n"
		writeBuf += 'PRECISION: ' + '\t' +  str(precision) + "\n"
		writeBuf += 'RECALL   : ' + '\t' +  str(recall) + "\n"
		writeBuf += 'F Score  : ' + '\t' +  str(fscore) + "\n"
		writeBuf += 'CONFUSION MATRIX: \n'
		writeBuf += str(model.confusion_matrices[key])
		writeBuf += '\n\n'

	if options['WRITE2FILE']:
		fileName = options['WRITE_DIR'] + 'Results_' + str(options['THRESHOLD']) + '_.txt'
		with open(fileName, 'wb') as f:
			f.write(writeBuf)
	else:
		print writeBuf
		sys.stdout.flush()

