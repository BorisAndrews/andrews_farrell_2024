'''
Mini package for handling options passed through terminal
'''
import sys

def get(option, type=None, default=None):
    argv = sys.argv
    option_string = "--" + option
    if option_string in argv:
        out = argv[argv.index(option_string) + 1]
        if type == None:
            return out
        else:
            return type(out)
    elif default == None:
        raise ValueError(option_string, "must be specified")
    else:
        return default