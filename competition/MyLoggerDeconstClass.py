
from logging import getLogger, StreamHandler, DEBUG
import logging 


class MyLoggerDeconstClass(object):
    
    def __init__(self):
        self.name = "myloggerdeconstructor"
        self.log = getLogger('root')

        
    def __del__(self):
        
        #
        # this is necessary process to delete current message handler 
        # for logging.....
        #
        root = self.log
        map(root.removeHandler, root.handlers[:])
        map(root.removeFilter, root.filters[:])

    
    

