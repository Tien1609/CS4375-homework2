def accuracy(Y_predict: list, Y_true: list) -> float:
    '''
    Predicts the accuracy pf the files
    :param list
    :return accuracy
    '''
    # Gets size of predicted 
    size = len(Y_true)
    # Calculates how many instances were accurate
    numOfAccurate = sum([1 if Y_predict[i] == Y_true[i] else 0 for i in range(size)])

    return numOfAccurate / size