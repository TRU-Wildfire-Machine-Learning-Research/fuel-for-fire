from Utils.Misc import *

class Label(object):
    def __init__(self, name, bin):
        self.name = name
        self.label_false_value_dict = {
            'water' : 128,
            'shrub' : 0.0,
            'mixed' : 0.0,
            'conifer' : 0.0,
            'herb' : 0.0,
            'cutbl' : 0.0,
            'exposed' : 0.0
        }
        self.label_true_value_dict = {
            'river' : 1.0,
            'broadleaf' : 1.0
        }
        self.samples, self.lines, self.bands, self.RawLabel = read_binary(bin)
        self.__build_binary()

    def __build_binary(self):
        # if the class name is in the false dict
        if self.name in self.label_false_value_dict:
            self.Binary = self.RawLabel != self.label_false_value_dict[self.name]
        elif self.name in self.label_true_value_dict:
            self.Binary = self.RawLabel == self.label_true_value_dict[self.name]