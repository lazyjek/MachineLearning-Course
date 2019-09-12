#-*- coding: utf-8 -*-
import arff
import numpy as np

def data_provider(filename):
    """ load data from arff files.
    @param filename[string]: input filename directory. string. example: 'sonar.arff'
    @rparam attrNum[integer]: number of attributes.
    @rparam labels[list]: all possible outputs of labels. for binary classification, there's only two items.
    @rparam instances[[list]]: instances, [[x1, x2, x3 .. xn label], .. ]
    """
    data = arff.load(open(filename, 'rb'))
    # return feature and datasets
    attrNum = len(data['attributes']) - 1
    labels = data['attributes'][-1][1]
    instances = data['data']
    return attrNum, labels, instances

def normalize(x):
    xMax, xMin= x.max(axis = 0), x.min(axis = 0)
    return (x - xMin) / (xMax - xMin)

def data_parser(instances):
    """ parse instances into label vector and instance vector """
    instances = np.array(instances)
    # y are labels [class1, class2, class1, class1, ....]
    y = np.array(instances[:,-1], dtype = np.str)
    x = np.array(instances[:,:-1], dtype = np.float64)
    x = normalize(x)
    #ones = np.ones(x.shape[0])
    #x = np.c_[ones, x]
    return x, y

if __name__ == '__main__':
    import sys
    datas = data_provider(sys.argv[1])
    data_parser(datas[2])
    #print datas[0]
    #print datas[1]
