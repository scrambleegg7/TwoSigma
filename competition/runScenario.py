# -*- coding: utf-8 -*-

from DataClass import DataClass


def proc1():
    
    
    
    dataCls = DataClass()
    colnames = dataCls.getColumns()

    print colnames


    train_data = dataCls.getTrain()
    print train_data.head()


def main():
    proc1()
    




if __name__ == "__main__":
    main()