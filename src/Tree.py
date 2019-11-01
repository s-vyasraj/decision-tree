# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:52:45 2019

@author: vyas
"""
import numpy as np
from Data import *

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
        if (self.GetTotalNumInputColumn >= 1):
            for i in np.arange(self.total_number_of_column):
                column_obj = self.column_list[i]
                if (column_obj.GetTypeOfColumn() == "input"):
                    return "valid", i
        return "invalid", 0
    
    def GetFirstOutColumnIndex(self):
        if (self.GetTotalNumInputColumn >= 1):
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
        print(Tree.AddColumn.__name__, " type: ", column_obj.GetTypeOfColumn())

        if(column_obj.GetTypeOfColumn() == "output"):
            if (self.output_column_present == "true"):
                print("Error - More than 1 output column - Unhandled]\n")
                exit(0)
            self.output_column_num = self.total_number_of_column
            
        self.column_list.append(column_obj)
        self.total_number_of_column += 1
        
    
    def ComputeColumnEntropy(self, column_num):
        print(Tree.ComputeColumnEntropy.__name__, " column_num: ", column_num)

        output_obj = self.column_list[self.output_column_num]
        output_data = output_obj.GetColumnData()
        u_out, out_indices, out_counts = np.unique(output_data, return_index=True, return_counts=True)

        column_obj = self.column_list[column_num]
        column_data = column_obj.GetColumnData()
        u_inp, inp_indices, inp_counts = np.unique(column_data, return_index=True, return_counts=True)

        probability = []
        entropy = 0
        if (len(u_out) == 1):
            probability.append(1)
            return entropy, probability, u_out[0], "leaf-node"        
        else:
            for i in np.arange(len(inp_counts)):
                p = inp_counts[i]/sum(inp_counts)
                entropy += -1.0 * p * np.log2(p)
                probability.append(p)
            return entropy, probability, "", "middle-node"        
                
        
        
    def GetEntropy(self, column_num):
        print(Tree.GetEntropy.__name__, "column_num: ", column_num)

        column_obj = self.column_list[column_num]
        if (column_obj.GetTypeOfColumn() == "output"):
            return  0, "invalid"
        
        if(self.output_column_present == " false"):
            print("Error - Output column not present .. bailing out\n")
            exit(0)
        
        e,plist, value, node_type = self.ComputeColumnEntropy(column_num)
        return e, node_type
        
    def ComputeEntropy(self):
        print(Tree.ComputeEntropy.__name__)

        column_entropy=[]
        for i in np.arange(self.total_number_of_column):
            e, node_type = self.GetEntropy(i)
            if (node_type != "invalid"):
                column_entropy.append([i, self.GetEntropy(i)])
        column_entropy = np.array(column_entropy)
        print(column_entropy)
        self.elist = column_entropy[:,1]
        self.sorted_entropy_index = np.argsort(self.elist)
        print(self.elist)
        print(self.sorted_entropy_index)
    
    def GetMinEntropyNode(self):
        print(Tree.GetMinEntropyNode.__name__)
        return self.sorted_entropy_index[0]

    def GetNextRootNode(self):
        print(Tree.GetNextRootNode.__name__)
        
        if (self.is_get_next_valid == "true" ):
            self.get_next_sorted_column += 1
        else:
            self.is_get_next_valid = "true" #first time
        
        if (len(self.sorted_entropy_index) > self.get_next_sorted_column):
            return self.sorted_entropy_index[self.get_next_sorted_column], "valid"
        else:
            return "", "invalid"
    
    def GetColumn(self, index):
        self.column_list[index]
        
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
        in_col = self.GetFirstInputColumnIndex()
        out_col = self.GetFirstOutColumnIndex()
        out_col_obj = self.column_list[out_col]
        in_col_obj = self.column_list[in_col]
        u_count = out_col_obj.GetNumUniqueValueCount()
        if (u_count == 1):
            u_value = out_col_obj.GetUniqueData(0)
            print(" ", in_col_obj.GetName(), "[",u_value, "] Probability = ", 1)
        else:
            for i in np.arange(u_count):
                u_value = out_col_obj.GetUniqueData(i)
                index_values = out_col_obj.GetIndexValues(u_value)
                p,values = in_col_obj.GetSparseProbability(index_values)
                for j in np.arange(len(values)):
                    print(" ", in_col_obj.GetName(), "[",values[j], "] Probability = ", p[j])
        return


    
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
    
def PreOrderTraversal(t, valid_flag, depth):
    print("PreOrderTraversal: valid_flag: ", valid_flag, "depth:", depth)

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