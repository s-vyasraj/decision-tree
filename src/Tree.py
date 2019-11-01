# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:52:45 2019

@author: vyas
"""
import numpy as np
from Data import *

debug_tree = "true"

class Tree:
    def __init__ (self, depth):
        self.column_list = []
        self.total_number_of_column = 0 
        self.output_column_present = "false" 
        self.output_column_num =  0 #index 0
        self.entropy = []
        self.get_next_sorted_column = 0
        self.is_get_next_valid = "false" #invalid
        
    def GetFirstInputColumnIndex(self):
        if (self.GetTotalNumInputColumn() >= 1):
            for i in np.arange(self.total_number_of_column):
                column_obj = self.column_list[i]
                if (column_obj.GetTypeOfColumn() == "input"):
                    return "valid", i
        return "invalid", 0
    
    def GetFirstOutColumnIndex(self):
        if (self.GetTotalNumInputColumn() >= 1):
            for i in np.arange(self.total_number_of_column):
                column_obj = self.column_list[i]
                if (column_obj.GetTypeOfColumn() == "output"):
                    return "valid", i
        return "invalid", 0

                
    def Print(self):
        print("total number of column: ", self.total_number_of_column)
        print("output_column_present: ", self.output_column_present)
        print("output_column_num: ", self.output_column_num)
        print("get_next_sorted_column: ", self.get_next_sorted_column)
        print("is_get_next_valid: ", self.is_get_next_valid)
        print("Length of column list: ", len(self.column_list))
        
    def AddColumn(self, column_obj):
        #print(Tree.AddColumn.__name__, " type: ", column_obj.GetTypeOfColumn())

        if(column_obj.GetTypeOfColumn() == "output"):
            if (self.output_column_present == "true"):
                print("Error - More than 1 output column - Unhandled]\n")
                exit(0)
            self.output_column_num = self.total_number_of_column
            self.output_column_present = "true"
            
        self.column_list.append(column_obj)
        self.total_number_of_column += 1
        
    
    def ComputeEntropy(self):
        #print(Tree.ComputeEntropy.__name__, self.output_column_num)

        column_entropy=[]
        output_column = self.column_list[self.output_column_num]
        for i in np.arange(self.total_number_of_column):
            if (i != self.output_column_num):
                input_column = self.column_list[i]
                e = Entropy(input_column, output_column)
                column_entropy.append([i, input_column.GetName(), e])
        column_entropy = np.array(column_entropy)
        #print("...", column_entropy)
        self.elist = column_entropy[:,2]
        self.sorted_entropy_index = np.argsort(self.elist)
        #print(self.elist)
        #print(self.sorted_entropy_index)
    
    def GetMinEntropyNode(self):
        #print(Tree.GetMinEntropyNode.__name__, self.sorted_entropy_index)
        return self.sorted_entropy_index[0]

    def GetNextRootNode(self):
        #print(Tree.GetNextRootNode.__name__)
        
        if (self.is_get_next_valid == "true" ):
            self.get_next_sorted_column += 1
        else:
            self.is_get_next_valid = "true" #first time
        
        if (len(self.sorted_entropy_index) > self.get_next_sorted_column):
            return self.sorted_entropy_index[self.get_next_sorted_column], "valid"
        else:
            return "", "invalid"
    
    def GetColumn(self, index):
        #print(Tree.GetColumn.__name__, "col name: ", index)
        #print(Tree.GetColumn.__name__, self.column_list[index].GetName())
        #self.Print()
        return self.column_list[index]
        
    def GetTotalNumInputColumn(self):
        count = 0
        for i in np.arange(self.total_number_of_column):
            column_obj = self.column_list[i]
            if (column_obj.GetTypeOfColumn() == "input"):
                count +=1
        return count

        
    def GetTotalNumOutputColumn(self):
        count = 0
        for i in np.arange(self.total_number_of_column):
            column_obj = self.column_list[i]
            if (column_obj.GetTypeOfColumn() == "output"):
                count +=1
        return count
    
    def GetTotalColumn(self):
        return self.total_number_of_column
    
    def IsLeaf(self):
        if (self.GetTotalNumInputColumn == 1):
            return "true"
        else:
            return "false"
    
    def PrintLeafProbability(self):
        print(Tree.GetColumn.__name__)
        in_valid, in_col = self.GetFirstInputColumnIndex()
        out_valid, out_col = self.GetFirstOutColumnIndex()

        out_col_obj = self.column_list[out_col]
        in_col_obj = self.column_list[in_col]
        u_count = out_col_obj.GetNumUniqueValueCount()
        if (u_count == 1):
            u_value = out_col_obj.GetUniqueData(0)
            print(" ", in_col_obj.GetName(), "[",u_value, "] Probability = ", 1)
        else:
            #print("... u_count", u_count)
            #out_col_obj.Debug()
            for i in np.arange(u_count):
                u_value = out_col_obj.GetUniqueData(i)
                index_values = out_col_obj.GetIndexValues(u_value)
                p,values = in_col_obj.GetSparseProbability(index_values)
                for j in np.arange(len(values)):
                    print(" ", in_col_obj.GetName(), "[",values[j], "] Probability = ", p[j])
        return


    
def GetSubTree(depth, original_tree, remove_column_index, decision_value):
    #print("GetSubTree: remove_column_index: ", remove_column_index, "Decision: ", decision_value," depth:", depth)
    t = Tree(depth)
    column_obj = original_tree.GetColumn(remove_column_index)
    index_values = column_obj.GetIndexValues(decision_value)
    for i in np.arange(original_tree.GetTotalColumn()):
        if (i != remove_column_index):
            valid_column = original_tree.GetColumn(i)
            coltype, name, attr1, attr2, ncolumn = valid_column.GetPrunedColumn(index_values)
            #print("GetSubTree: remove_column_index: ", remove_column_index, "remove_name: ", column_obj.GetName(),
            #          "Decision: ", decision_value,
            #          " depth:", depth, " ncolumn: ", name , " values: ", ncolumn )

            c1 = ClassColumn(coltype, name, attr1, attr1, ncolumn)
            t.AddColumn(c1)
    
    #print("GetSubTree:...Total columns in new tree:", t.GetTotalNumInputColumn())
    
    if (t.GetTotalNumInputColumn() <= 1):
        return t, "leaf"
    return t, "mid-node"

def PrintLevel(level):
    for i in np.arange(level):
        print(".", end = " ")         
        
def PreOrderTraversal(t, valid_flag, depth):
    #print("PreOrderTraversal: valid_flag: ", valid_flag, "depth:", depth)

    #t.Print()
    t.ComputeEntropy()
    PrintLevel(depth)

    if (valid_flag == "valid"):
        col_num = t.GetMinEntropyNode()
        column_obj = t.GetColumn(col_num)

        unique_decision_count = column_obj.GetNumUniqueValueCount()
        
        for i in np.arange(unique_decision_count):
            filter_data = column_obj.GetUniqueData(i)
            new_tree, tree_type = GetSubTree(depth+1, t, col_num, filter_data)
            if (tree_type == "mid-node"):
                print(column_obj.GetUserQuestion(i))
                PreOrderTraversal(new_tree, "valid", depth+1)
            else:
                print("..leaf..")
                new_tree.PrintLeafProbability()       

def Test():
    i1= [ "rainy", "rainy", "overcast", "sunny", "sunny", "sunny", "overcast", "rainy", "rainy", "sunny", "rainy", "overcast", "overcast", "sunny"]
    i2 = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot","mild"]
    i3 = ["high", "high", "high", "high", "normal", "normal", "normal", "high","normal", "normal", "normal", "high", "normal", "high" ]
    i4 = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true","false", "true" ]
    o1 = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
    #print(len(i1), len(i2), len(i3), len(i4), len(o1))
    c1 = ClassColumn("input", "Weather" , "rainy", "sunny", i1)
    c2 = ClassColumn("input", "Temperature" , "rainy", "sunny", i2)
    c3 = ClassColumn("input", "Humidity" , "rainy", "sunny", i3)
    c4 = ClassColumn("input", "Windy" , "rainy", "sunny", i4)
    co = ClassColumn("output", "Play" , "yes", "no", o1)

    #print("Get input Probability: ", ci.GetProbability())
    #print("Get output Probability: ", co.GetProbability())
    t = Tree(0)
    t.AddColumn(c1)
    t.AddColumn(c2)
    t.AddColumn(c3)
    t.AddColumn(c4)
    
    t.AddColumn(co)
    t.ComputeEntropy()
    PreOrderTraversal(t,"valid", 0)

    return

if __name__ == "__main__":
    # execute only if run as a script
    Test()
