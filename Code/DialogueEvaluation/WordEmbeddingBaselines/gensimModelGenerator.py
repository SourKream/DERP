#TODO : Take care of imports
import gensim
import sys
import progressbar
#TODO : change tokenizer

def tokenize(data):
	return data.split(' ')

class Generator(object):
	def __init__(self, dir_file):
		self.file_name = dir_file
		self.epoch_number = 0
		self.counter = 0
		self.PRINT_THRESH = 2
		self.bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

	
	def __iter__(self):
		self.epoch_number += 1
		print 'STARTING EPOCH ', self.epoch_number
		self.counter = 0
		with open(self.file_name) as f:
			for line in f:
				self.counter += 1
				self.bar.update(self.counter)
				yield tokenize(line)
	
if __name__ == "__main__":
	local = True
	if local :
		data_dir = '../../../'
		model_dir = "../../../"
		data_file = data_dir + 'DataInRawTextFormat.txt'
		model_file = model_dir + 'gensim_model.model'
	else:
		data_dir = "/home/cse/dual/cs5130275/scratch/DERP/Reddit/DatasetNewPruned11M/"
		model_dir = "/home/cse/dual/cs5130275/scratch/DERP/Models/Gensim_Models/"
		data_file = data_dir + 'DataInRawTextFormat.txt'
		model_file = model_dir + 'gensim_model.model'
	generator = Generator(data_file)
	model = gensim.models.Word2Vec(generator, min_count = 10, iter= 10, size=200, workers=4)
	model.save(model_file)

