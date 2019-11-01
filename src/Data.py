# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:38:43 2019

@author: vyas
Manages the input data / output result .. housekeeping etc.,
Y = hypothesis(X)
"""


import csv
import codecs
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


        
    
def GetSubTree(depth, original_tree, remove_column_index, decision_value):
    t = Tree(depth)
    column_obj = original_tree.GetColumn(remove_column_index)
    index_values = column_obj.GetIndexValues(decision_value)
    for i in np.arange(original_tree.GetTotalColumn()):
        if (i != remove_column_index):
            valid_column = t.GetColumn(i)
            coltype, name, attr1, attr2, ncolumn = valid_column.GetPrunedColumn(index_values)
            c1 = ClassColumn(coltype, name, attr1, attr1, ncolumn)
            t.AddColumn(c1)
    
    if (t.GetTotalNumInputColumn <= 1):
        return t, "leaf"
    return t, "mid-node"

     

def ImportData(file_name):
    print(__name__)
    data=[]
    print("Importing data....")
    
#a1 Temperature of patient { 35C-42C }
#a2 Occurrence of nausea { yes, no }
#a3 Lumbar pain { yes, no }
#a4 Urine pushing (continuous need for urination) { yes, no }
#a5 Micturition pains { yes, no }
#a6 Burning of urethra, itch, swelling of urethra outlet { yes, no }
#d1 decision: Inflammation of urinary bladder { yes, no }
#d2 decision: Nephritis of renal pelvis origin { yes, no }
    
    #TBD:Hard coding today
    # ColumnNum, "Name", numder of Decision branch, values of decision branch
   
      
    f=codecs.open(file_name,"rb","utf-16")
    csvread=csv.reader(f,delimiter=',')
    for row in csvread:
        data.append(row)
    ndata = np.asarray(data)
    c1 = ClassColumn("input", "Temperature", "less_than_38", "greater_than_38", ndata[:,0])
    c2 = ClassColumn("input", "Nausea", "yes", "no", ndata[:,1])
    c3 = ClassColumn("input", "Lumbar", "yes", "no", ndata[:,2])
    c4 = ClassColumn("input", "Urine pushing", "yes", "no", ndata[:,3])
    c5 = ClassColumn("input", "Micturition", "yes", "no", ndata[:,4])
    c6 = ClassColumn("input", "Burning", "yes", "no", ndata[:,5])
    c7 = ClassColumn("output", "Decision: Inflamation", "yes", "no", ndata[:,6])
    
    t = Tree(0)
    t.AddColumn(c1)
    t.AddColumn(c2)
    t.AddColumn(c3)
    t.AddColumn(c4)
    t.AddColumn(c5)
    t.AddColumn(c6)
    t.AddColumn(c7)

    return t, ndata

def PrepareData(header, ndata):
    return ndata

def PrintLeafResult(t1):
    total_input_column = t1.GetTotalNumInputColumn()
    if (total_input_column == 1):
        for i in np.arange(t1.GetTotalColumn):
            column_obj = t1.GetColumn(i)
            if (column_obj.GetTypeOfColumn() == "output"):
                if (column_obj.GetNumUniqueValueCount == 1):
                    column_obj.PrintProbability()
      
def PreOrderTraversal(t, valid_flag, depth):
    t.Print()
    t.ComputeEntropy()
    
    if (valid_flag == "valid"):
        col_num = t.GetMinEntropyNode()
        column_obj = t.GetColumn(col_num)
        print("...", column_obj.GetName())
        unique_decision_count = column_obj.GetNumUniqueValueCount()
        
        for i in np.arange(unique_decision_count):
            filter_data = column_obj.GetUniqueData(i)
            new_tree, tree_type = GetSubTree(depth+1, t, col_num, filter_data)
            if (tree_type != "mid-node"):
                PreOrderTraversal(new_tree, "valid", depth+1)
            else:
                t.PrintLeafProbability()
                
    
def Dtree(t):
    print(__name__)
    PreOrderTraversal(t,"valid", 0)
          
        
def main():
    file_name = "G:/code/decision-tree/data/nephra/diagnosis.clean.data"
    tree, ndata = ImportData(file_name)
    Dtree(tree)

    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()