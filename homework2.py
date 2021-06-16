from fileReader import getAttributeNames, getAttributeNames, getAttributeValues, getClassValues, getFileName, readData, readFiles
import os
import sys

def main():
    '''
    Main function that runs all ther different functions
    :param None
    :return None
    '''
    # Gets the training and test files
    (trainFileName, testFileName) = getFileName(sys.argv)
    # Reads the files
    trainData = readData(trainFileName)
    testData = readData(testFileName)
    # Retrieves the attribute values
    X = getAttributeValues(trainData)
    # Retrieves the class column from the training file
    Y = getClassValues(trainData)
    # Retrieves the attribute names
    M = getAttributeNames(trainData)
    # Gets the attribute values from test file
    XTest = getAttributeValues(testData)
    # Retrieves the class column from the test file
    YTest = getClassValues(testData)
    
    # X: data of instances
    print("Data from X:")
    print(X)
    
    # Y: data of class variables
    print("Data from Y:")
    print(Y)
    
    # M: data of attribute names
    print("Data from M:")
    print(M)
    

if __name__ == "__main__":
    main()
