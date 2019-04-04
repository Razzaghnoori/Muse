import numpy as np
import logging

from os.path import join, exists, isdir, isfile
from os import listdir, walk


class Dirohe(object):
    """
    Encode each file and directory inside the root_dir to a
    one-hot encoding representation.
    """

    def __init__(self, root_dir, encodings=None, \
        filter_=None):

        self.filters = {
            'dirs': lambda x: isdir(x),
            'midis': lambda x: x.split('.')[-1] == 'mid',
            'files': lambda x: isfile(x),
            None: lambda x: True
        }

        self.selected_filter = self.filters[filter_]
        self.root_dir = root_dir
        self.q = list()
        self.paths = []
        self.ohe = dict()
        self.n_nodes_in_depth = dict()
        
        if encodings is None:
            self.start()
            self.end()
        else:
            self.encodings = encodings
            self.ohe = dict(
                zip(self.get_path_list(),
                self.encodings.tolist())
            )

    def get_path_list(self):
        if self.paths:
            return self.paths
        
        self.paths = []
        for dir, _, files in walk(self.root_dir):
            if self.selected_filter(dir):
                self.paths.append(dir.rstrip('/'))
            self.paths.extend([join(dir, x.rstrip('/')) for x in files \
                if self.selected_filter(join(dir, x))])
        
        return self.paths

    def start(self):
        self.q.append(self.root_dir.rstrip('/'))
        self.ohe[''] = []    #Father of all fathers (root) should have a father.
        self.process(0)
    
    def process(self, depth):
        if not self.q:
            return
        
        nodes_in_depth = self.q
        self.n_nodes_in_depth[depth] = len(self.q)
        self.q = []

        encodings = np.eye(len(nodes_in_depth), dtype=int).tolist()
        self.ohe.update(dict(zip(nodes_in_depth, encodings)))

        for node in nodes_in_depth:
            if not isdir(node):
                continue
            children = [join(node, x.rstrip('/')) for x in listdir(node) \
                if self.selected_filter(join(node, x))]
            self.q.extend(children)

        self.process(depth+1)

    def join_encodings(self, me, dad):
        me = me.rstrip('/')
        dad = dad.rstrip('/')
        self.ohe[me] = self.ohe[dad] + self.ohe[me]

        if isdir(me):
            for child in listdir(me):
                child = join(me, child)
                if not self.selected_filter(child):
                    continue
                self.join_encodings(child, me)

    def normalize_encodings(self):
        max_len = max([len(self.ohe[k]) for k in self.ohe])
        
        self.dim = max_len

        for k in self.ohe:
            self.ohe[k] += (max_len - len(self.ohe[k])) * [0]

    def create_encoding_matrix(self):
        if hasattr(self, 'encodings'):
            return
        
        self.encodings = []
        for path in self.get_path_list():
            ohe = self.ohe.get(path)
            if not ohe:
                #TODO: Replace this with logging.log
                print('Warning: path {} has no calculated one-hot encoding available.'.format(path))
                continue
            self.encodings.append(ohe)
        
        self.encodings = np.array(self.encodings)

    def export(self, exp_dir):
        try:
            np.save(exp_dir, self.encodings)
            print('Encodings saved successfully.')
        except:
            print("Couldn't save encodings. Please try again.")
    
    def end(self):
        self.join_encodings(self.root_dir, '')
        self.normalize_encodings()
        self.create_encoding_matrix()

    def encode(self, path):
        while True:
            enc = self.ohe.get(path)
            if enc is not None:
                return enc

            sep_ind = path.rfind('/')
            if sep_ind == -1:
                return None

            path = path[:sep_ind]
        
    def get_random_encodings(self, shape):
        size = np.prod(shape)
        inds = np.random.randint(0, len(self.ohe), size)
        encodings = self.encodings[inds]
        return encodings.reshape(shape)





