from Utils.Misc import *

class Label(object):
    def __init__(self, name, bin):
        self.name = name
        self.samples, self.lines, self.bands, self.Data = read_binary(bin)

