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
        self.entropy = 0
        for i in np.arange(len(self.unique_counts)):
            p = self.unique_counts[i]/sum(self.unique_counts)
            self.probability.append(p)
            self.entropy += -1.0 * p *np.log2(p)
            
        if (len(self.probability) == 1):
            self.is_redundant = "true"   # This column is not conveying anything
        
        self.questions= []
        for i in self.unique_values:
            if (self.type_of_column == "input"):
                str_var = " Is " + self.name + " " + i +  " ? "
            else: #output
                str_var = self.name + " = " + i
            self.questions.append( str_var )
            

    def GetUserQuestion(self, index):
        return self.questions[index]
    
    def GetUserQuestions(self):
        return self.questions
    
    
    def Debug(self):
        print("Name: ", self.name)
        print("unique_values: ", self.unique_values)
        print("unique_indices: ", self.unique_indices)
        print("unique_counts: ", self.unique_counts)
        print("Entropy: ", self.entropy)
        
    def PrintProbability(self):
        for i in np.range(len(self.unique_values)):
            print(" ", self.name, "[",self.unique_values[i], "] Probability = ", self.probability[i])

    def GetNumUniqueValueCount(self):
        return len(self.unique_values)
    
    # if unique_index = 0, returns first unique value
    #....
    def GetUniqueData(self, index):
        idx = self.unique_indices[index]
        #print("....", idx, self.data[0][0])
        return self.data[idx]
    
    def GetIndexValues(self, value):
        #print("..", value, "... " , self.data)
        d = np.array(self.data)
        index_values = np.argwhere(d == value)
        #print("...", index_values)
        return index_values
    
    def GetPrunedColumn(self, index_values):
        new_column = np.delete(self.data, index_values)
        return self.type_of_column, self.name, self.attribute_value1, self.attribute_value2, new_column
    
    def GetFilterColumn(self, index_values):
        d=np.array(self.data)
        f = d[index_values] 
        l = f.shape[0]
        fnew = f.reshape(1,l)[0]
        return self.type_of_column, self.name, self.attribute_value1, self.attribute_value2, fnew
    
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
    
    def GetProbability(self):
        return self.probability, self.unique_values
    
    def GetColumnEntropy(self):
        return self.entropy
 
# Takes 2 column and calculates Entropy
def Entropy(input_column, result_column):
   
    in_p,in_unique_val = input_column.GetProbability()
    entropy = 0
    for i in np.arange(len(in_unique_val)):
        in_attr_value = in_unique_val[i]
        #print(in_attr_value)
        index_values = input_column.GetIndexValues(in_attr_value)
        #print(index_values)
        coltype, name, attr1, attr2, ncolumn = result_column.GetFilterColumn(index_values)
        nc = ClassColumn(coltype, name, attr1, attr2, ncolumn)
        entropy += in_p[i] * nc.GetColumnEntropy()
        #print ("i: ", i, "attr:", in_attr_value, "data:", ncolumn)
        #print("i: ", i, " Entropy: ", nc.GetColumnEntropy())
    
    return entropy

def Test():
    i1= [ "rainy", "rainy", "overcast", "sunny", "sunny", "sunny", "overcast", "rainy", "rainy", "sunny", "rainy", "overcast", "overcast", "sunny"]
    i2 = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot","mild"]
    i3 = ["high", "high", "high", "high", "normal", "normal", "normal", "high","normal", "normal", "normal", "high", "normal", "high" ]
    i4 = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true","false", "true" ]
    o1 = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
    print(len(i1), len(i2), len(i3), len(i4), len(o1))
    c1 = ClassColumn("input", "Weather" , "rainy", "sunny", i1)
    c2 = ClassColumn("input", "Temperature" , "rainy", "sunny", i2)
    c3 = ClassColumn("input", "Humidity" , "rainy", "sunny", i3)
    c4 = ClassColumn("input", "Windy" , "rainy", "sunny", i4)

    co = ClassColumn("output", "Play" , "yes", "no", o1)
    print("Get input Probability: ", c1.GetProbability())
    print("Get output Probability: ", co.GetProbability())
    e = Entropy(c1, co)
    print("Final entropy:", e)
    print(c1.GetUserQuestion(0))
    print(c1.GetUserQuestion(1))
    print(co.GetUserQuestion(1))

    r = c1.GetIndexValues("rainy")
    print(r)
    return

if __name__ == "__main__":
    # execute only if run as a script
    Test()

