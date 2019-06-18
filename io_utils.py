import pickle


def dump_pickle(content, path, mode='wb'):
    with open(path, mode) as f:
        pickle.dump(content, f)


def load_pickle(path):
    if 'streaming' in path:
        return load_streaming_pickle(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_streaming_pickle(path):
    items = []
    with open(path, 'rb') as f:
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break
            items += [item]
    return items
