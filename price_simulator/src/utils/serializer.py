import pickle


def load_object(filename):
    """ Unpickle a file of pickled data. """
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_multiple_objects(filename):
    """ Stream a file of pickled objects. """
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def save_object(obj, filename):
    """ Pickle a python object to filename.pkl . """
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def add_object(obj, filename):
    """ Add a python object to filename.pkl . """
    with open(filename, "ab") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
