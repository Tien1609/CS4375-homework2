#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import os
'''
This class reads the files from the command line
@author Tien Duong and Julie Park
'''
def getFileName(args):
    '''
    Gets the files and checks if the input is valid
    :param arguments
    :return argument[1] and argument[2] (strings)
    '''
    # If length is not 3 will output the following message
    if len(args) != 3:
        print("<usage>: {0} <training_file> <testing_file>".format(args[0]))
        print("         ex) [0] train.dat test.dat".format(args[0]))
        sys.exit(0)
    return (args[1], args[2])

def readFiles(fileName):
    '''
    Reads the files and checks if the input is valid
    :param fileName (string)
    :return string
    '''
    lines = None
    with open(fileName, 'r') as fd:
        lines = fd.readlines()
    return lines

def readData(fileName):
    '''
    Reads the data within the file and checks if the input is valid
    :param fileName (string)
    :return list
    '''
    # Reads the file
    lines = readFiles(fileName)
    # Output error message if line is less than 2
    if lines is None or len(lines) < 2:
        raise ValueError("ERROR: invalid data file")

    fileData = []
    k = 0
    # Loops to retrieve each data to store in a list
    for l in lines:
        # Get attribute names as string
        row = l.strip().rstrip()
        if len(row) < 1:
            continue
 
        row = row.split()
        # Append the data into the list
        if k != 0:
        	row = [int(r) for r in row]
        fileData.append(row)
        k += 1
    
    attributeSize = len(fileData[0])
    for d in fileData:
        # If the size do not macth output error message
        if len(d) != attributeSize:
            raise ValueError("ERROR: columns mismatched")
    return fileData

def getAttributeValues(fileData):
    '''
    Gets the attribute values/data from the file
    :param fileData: contains the train file information
    :return: list of values
    '''
    # Gets all the data except the last column
    X = [row[:-1] for row in fileData[1:]]
    return X

def getClassValues(fileData):
    '''
    Gets the class values/data from the file
    :param fileData: contains the train file information
    :return: list of values
    '''
    # Gets the last column (class column)
    Y = [row[-1] for row in fileData[1:]]
    return Y

def getAttributeNames(fileData):
    '''
    Gets the attribute names from the file
    :param fileData: contains the train file information
    :return: list of attribute names
    '''
    # Retrieves the first row except the class column
    M = fileData[0][:-1]
    return M