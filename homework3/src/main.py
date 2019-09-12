import sys
from data_provider import data_provider
from cv import CrossValidate
from perceptron import Perceptron

if __name__ == '__main__':
    try:
        trainFileName = sys.argv[1]
        nfolds = int(sys.argv[2])
        learningRate = float(sys.argv[3])
        epochNum = int(sys.argv[4])
    except:
        print >> sys.stderr, '[ERROR] wrong input format! 4 inputs: [string] [int] [float] [int]'
    attrNum, labels, instances = data_provider(trainFileName)
    cv = CrossValidate(nfolds, instances, labels)

    """
    output: fold_of_instance | predicted_class | actual_class | confidence_of_prediction
    """
    perceptron = Perceptron(attrNum, labels, learningRate, epochNum)
    # nfolds training
    for i in range(nfolds):
        train, test = cv.fold(i)
        perceptron.fit(train[0], train[1])
        confidences = perceptron.confidence(test[0])
        predict = perceptron.predict(test[0])
        for j in range(test[0].shape[0]):
            print i, predict[j][0], test[1][j], '%6f'%confidences[j]




