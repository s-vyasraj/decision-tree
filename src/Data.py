# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:38:43 2019

@author: vyas
Manages the input data / output result .. housekeeping etc.,
Y = hypothesis(X)
"""



import numpy as np

class ClassColumn:
    #type_of_column = input output
    def __init__ (self, type_of_column, name, attribute_value1, attribute_value2, data):
        self.type_of_column = type_of_column
        self.name = name
        self.attribute_value1 = attribute_value1
        self.attribute_value2 = attribute_value2
        self.data = data
        self.unique_values, self.unique_indices, self.unique_counts =  np.unique(data, 
                            return_index=True, return_counts=True)
        self.probability = []
        for i in np.arange(len(self.unique_counts)):
            p = self.unique_counts[i]/sum(self.unique_counts)
            self.probability.append(p)
        if (len(self.probability) == 1):
            self.is_redundant = "true"   # This column is not conveying anything
    
    def PrintProbability(self):
        for i in np.range(len(self.unique_values)):
            print(" ", self.name, "[",self.unique_values[i], "] Probability = ", self.probability[i])

    def GetNumUniqueValueCount(self):
        return len(self.unique_values)
    
    # if unique_index = 0, returns first unique value
    #....
    def GetUniqueData(self, index):
        return self.data[self.unique_indices[index]]
    
    def GetIndexValues(self, value):
        index_values = np.argwhere(self.data == value)
        return index_values
    
    def GetPrunedColumn(self, index_values):
        new_column = np.delete(self.data, index_values)
        return self.type_of_column, self.name, self.attribute_value1, self.attribute_value2, new_column
    
    def GetTypeOfColumn(self):
        return self.type_of_column
    
    def GetColumnData(self):
        return self.data
    
    def GetName(self):
        return self.name
    
    def GetSparseProbability(self, index_list):
        new_list = []
        for i in index_list:
            new_list.append(self.data[i])
        new_list = np.array(new_list)
        unique_values, unique_indices, unique_counts =  np.unique(new_list, 
                            return_index=True, return_counts=True)
        probability = []
        for i in np.arange(len(unique_counts)):
            p = unique_counts[i]/sum(unique_counts)
            probability.append(p)
        return probability, unique_values


