import numpy as np
from my_utils import my_tokenize
from scipy import spatial
import gensim
import pdb
import sys

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
	options['THRESHOLD'] = 0.5
	if options['LOCAL']:
		if options['GLOVE']:
			options['MODEL_DIR'] = '/Users/barunpatra/Documents/glove.840B.300d.txt'
		else:
			options['MODEL_DIR'] = '../../../gensim_model.model'
	else:
		#TODO : Set HPC settings
		if options['GLOVE']:
			options['MODEL_DIR'] = '/home/cse/dual/cs5130275/scratch/DERP/glove.840B.300d.txt'
		else:
			options['MODEL_DIR'] = '/home/cse/dual/cs5130275/scratch/DERP/Models/Gensim_Models/gensim_model.model'
			
	return options

class BaseLineModels(object):
	def __init__(self, options):
		self.options = options
		self.confusion_matrices = {}				
		for method in options['EMBEDDING_METHODS']:
			self.confusion_matrices[method] = np.zeros((2,2))		
		print 'STARTING TO LOAD THE MODEL'
		sys.stdout.flush()
		self.setModel()
		print 'FINISHED LOADING MODEL'
		sys.stdout.flush()
	
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
		PRINT_THRESHOLD = 10
		with open(self.options['DATA_FILE']) as f:
			for line in f:				
				line = line.strip().split('\t')				
				if options['USE_CONTEXT']:
					# TODO : Fill this out
					pass
				else:					
					for method in self.options['EMBEDDING_METHODS']:						
						self.updateConfusionMatrix(method, my_tokenize(line[1]), my_tokenize(line[2]), eval(line[3]))
					counter += 1
					if (counter % PRINT_THRESHOLD) == 0:
						print counter, 'DONE'
						sys.stdout.flush()

	def getGreedy(self,seq1,seq2):
		count = 0
		G = 2.0
		for word1 in seq1:
			if word1 in self.model:
				embed_word1 = self.model[word1]
				cosine_dist = 2.0
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
		if method == 'GREEDY_MATCHING':		
			G_predicted_gold = self.getGreedy(predicted, gold)
			G_gold_predicted = self.getGreedy(gold, predicted)
			G_score = (G_gold_predicted + G_predicted_gold) / 2
			if G_score < self.options['THRESHOLD']:
				predicted_label = 1
			else:
				predicted_label = 0			

		elif method == 'EMBEDDING_AVERAGE':	
			predicted_embedding = self.getSentenceEmbedding(predicted)
			gold_embedding = self.getSentenceEmbedding(gold)
			if np.count_nonzero(predicted_embedding) == 0 or np.count_nonzero(gold_embedding) == 0:
				predicted_label = 0
			else:
				cosine_dist = spatial.distance.cosine(predicted_embedding, gold_embedding)
				if cosine_dist < self.options['THRESHOLD']:
					predicted_label = 1
				else:
					predicted_label = 0				

		elif method == 'VECTOR_EXTREMA':
			predicted_embedding = self.getSentenceEmbedding(predicted, False)
			gold_embedding = self.getSentenceEmbedding(gold, False)
			if np.count_nonzero(predicted_embedding) == 0 or np.count_nonzero(gold_embedding) == 0:
				predicted_label = 0
			else:
				cosine_dist = spatial.distance.cosine(predicted_embedding, gold_embedding)
				if cosine_dist < self.options['THRESHOLD']:
					predicted_label = 1
				else:
					predicted_label = 0	
		self.confusion_matrices[method][gold_label][predicted_label] += 1

if __name__ == "__main__":
	options = getParams()
	model = BaseLineModels(options)
	model.generateConfusionMatrices()
	for key in model.confusion_matrices:		
		print 'METHOD: ',key
		sys.stdout.flush()
		print 'ACCURACY: ', model.confusion_matrices[key][1][1] / model.confusion_matrices[key].sum() * 100.
		print 'CONFUSION MATRIX: '
		print model.confusion_matrices[key]
		print ''
		sys.stdout.flush()

