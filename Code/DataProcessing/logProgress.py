import sys

def logProgress(sequence, every=None, size=None, name='Items'):

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    index = 0
    for index, record in enumerate(sequence, 1):
        if index == 1 or index % every == 0:
            if is_iterator:
            	sys.stdout.write("Loading: " + str(index) + '\r')
            	sys.stdout.flush()
            else:
            	sys.stdout.write("Loading: " + str(index) + " / " + str(size) + '\r')
            	sys.stdout.flush()
        yield record
