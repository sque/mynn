from collections import namedtuple

LayerValues = namedtuple('LayerValues', ['Z', 'A', 'extras'])
LayerParameters = namedtuple('LayerParameters', ['W', 'b'])
LayerGrads = namedtuple('LayerGrads', ['dW', 'db'])
