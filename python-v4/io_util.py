""" io utility functions """

import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def safe_open_wb(path):
    mkdir_p(os.path.dirname(path))
    return open(path, 'wb')
