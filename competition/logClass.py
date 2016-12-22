# -*- coding: utf-8 -*-



from logging import getLogger, StreamHandler, DEBUG
from logging import Formatter
import logging 


class SubClass(object):
    
    def __init__(self):
        self.log = getLogger('root')
        
        self.log.info("from subclass")
    
    def doSomething(self):
        self.log.info("from subclass doSomething ")

class myLogger(object):
    
    def __init__(self):
        self.logger = getLogger(__name__)
        
        handler = StreamHandler()

        handler.setLevel(DEBUG)

        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)
        self.logger.setLevel(DEBUG)

        self.logger.addHandler(handler)
        
        
    def __del__(self):
        root = self.logger
        map(root.removeHandler, root.handlers[:])
        map(root.removeFilter, root.filters[:])
    
    def __exit__(self):
        print "exit"
        
    def debug(self,instr):
        self.logger.debug(instr)



class MyHandler(StreamHandler):

    def __init__(self):
        StreamHandler.__init__(self)
        fmt = '%(asctime)s %(filename)-18s %(levelname)-8s: %(message)s'
        fmt_date = '%Y-%m-%dT%T%Z'
        formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(formatter)



def main():
    
    
    log = getLogger("root")
    log.setLevel(DEBUG)
    log.addHandler(MyHandler())

    sub = SubClass()
    sub.doSomething() 
    
if __name__ == "__main__":
    main()
