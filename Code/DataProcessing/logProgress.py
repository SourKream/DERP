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
