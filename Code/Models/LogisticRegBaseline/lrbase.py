import pdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix

TRAIN_SIZE = 100
train_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/train.txt'
val_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/val.txt'
test_file = '/scratch/cse/dual/cs5130275/DERP/Reddit/DatasetWithPruning7M/test.txt'

def load_data(filename, num_dat_points=-1):
    f = open(filename, 'rt')
    
    if num_dat_points==-1:
        dat = f.readlines()
    else:
        dat = []
        for i in range(num_dat_points):
            dat.append(f.readline().strip())
            
    f.close()
    
    dat = [x.split('\t') for x in dat]
    assert(all([len(x)==4 for x in dat]))
    dat_x = [x[:3] for x in dat]
    dat_y = [int(x[3]) for x in dat]
    
    return dat_x, dat_y

def count_vec_tfidf(train_x):
    train_x = [y for x in train_x for y in x]   # flatten [[cxt,r1,r2], ...]

    count_vect = CountVectorizer(min_df=2, max_features=50000)
    count_vect.fit(train_x)
    train_x = count_vect.transform(train_x)
    print('done count vectorize fit')
    print('vocab size: ' + str(len(count_vect.get_feature_names())))
    
    tfidf_transformer = TfidfTransformer(use_idf=True)
    tfidf_transformer.fit(train_x)
    train_x = tfidf_transformer.transform(train_x)
    print('done tfidf fit')
    
    train_x = train_x.tolil().reshape((train_x.shape[0]/3,train_x.shape[1]*3))
    print('done count_vec_tfidf')
    
    return train_x, count_vect, tfidf_transformer

def test(test_x, count_vect, tfidf_transformer, clf, test_y, print_details=False):
    print_test = test_x
    test_x = [y for x in test_x for y in x]
    test_x = count_vect.transform(test_x)
    test_x = tfidf_transformer.transform(test_x)
    test_x = test_x.tolil().reshape((test_x.shape[0]/3, test_x.shape[1]*3))
    
    pred = clf.predict(test_x)
    
    if print_details:
        for i in range(len(pred)):
            print (test_y[i],pred[i],print_test[i])
            
    print(confusion_matrix(test_y,pred))    
    print("Accuracy: " + str(np.mean(pred == test_y)))
    

# MAIN
if __name__ == '__main__':
    
    train_x, train_y = load_data(train_file, TRAIN_SIZE)
    
    train_x, count_vect, tfidf_transformer = count_vec_tfidf(train_x)
    
    clf = LogisticRegression().fit(train_x, train_y)
    print('\ndone training')
    
    val_x, val_y = load_data(val_file,100)
    test_x, test_y = load_data(test_file,100)
    
    test(val_x, count_vect, tfidf_transformer, clf, val_y, print_details=True)
    test(test_x, count_vect, tfidf_transformer, clf, test_y, print_details=True)
    
    