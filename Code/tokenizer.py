import re
import pdb

tokenizer_exceptions = [u'(amazon\\.com)', u'(google\\.com)', u'(a\\.k\\.a\\.)', u'(r\\.i\\.p\\.)', u'(states\\.)', u'(a\\.k\\.a)', u'(r\\.i\\.p)', u'(corps\\.)', u'(ph\\.d\\.)', u'(corp\\.)', u'(j\\.r\\.)', u'(b\\.s\\.)', u'(alex\\.)', u'(d\\.c\\.)', u'(b\\.c\\.)', u'(bros\\.)', u'(j\\.j\\.)', u'(mins\\.)', u'(\\.\\.\\.)', u'(dept\\.)', u'(a\\.i\\.)', u'(u\\.k\\.)', u'(c\\.k\\.)', u'(p\\.m\\.)', u'(reps\\.)', u'(prof\\.)', u'(p\\.s\\.)', u'(l\\.a\\.)', u'(i\\.e\\.)', u'(govt\\.)', u'(u\\.s\\.)', u'(t\\.v\\.)', u'(a\\.m\\.)', u'(cons\\.)', u'(e\\.g\\.)', u'(j\\.k\\.)', u'(ave\\.)', u'(gen\\.)', u'(feb\\.)', u'(mrs\\.)', u'(etc\\.)', u'(vol\\.)', u'(gov\\.)', u'(sec\\.)', u'(nov\\.)', u'(hrs\\.)', u'(sgt\\.)', u'(mon\\.)', u'(jan\\.)', u'(min\\.)', u'(pts\\.)', u'(rev\\.)', u'(inc\\.)', u'(est\\.)', u'(cal\\.)', u'(sat\\.)', u'(dec\\.)', u'(rep\\.)', u'(lbs\\.)', u'(mr\\.)', u'(jr\\.)', u'(km\\.)', u'(dc\\.)', u'(p\\.s)', u'(pp\\.)', u'(ex\\.)', u'(op\\.)', u'(co\\.)', u'(sp\\.)', u'(u\\.s)', u'(vs\\.)', u'(kg\\.)', u'(ms\\.)', u'(iv\\.)', u'(ca\\.)', u'(sr\\.)', u'(oz\\.)', u'(bc\\.)', u'(dr\\.)', u'(ga\\.)', u'(lb\\.)', u'(mi\\.)', u'(ad\\.)', u'(ft\\.)', u'(e\\.g)', u'(ed\\.)', u'(sc\\.)', u'(lt\\.)', u'(va\\.)', u'(la\\.)', u'(mt\\.)', u'(i\\.e)', u'(st\\.)', u'(mo\\.)']

def my_tokenize(sent):
    return [''.join(x) for x in re.findall("|".join(tokenizer_exceptions)+"|([0-9]+)|('\w{1,2}[^\w])|([\w]+)|([.,!?;'])",sent)]

def map_to_ix(word, vocab, lower=True):
    '''
        inputs:
            word : the word
            vocab: a dictionary mapping words to indices. Is zero indexed (i.e <MASK_TOK> is not present) and doesn't 
                    contain an index for <UNK>
            lower: convert word to lowercase
        output:
            the index of the word (+1 for accommodating the masking token)
    '''
    if lower:
        word = word.lower()
    if word in vocab:
        return vocab[word] + 1
    else:
        return len(vocab) + 1