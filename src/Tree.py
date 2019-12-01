# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:52:45 2019

@author: vyas
"""
import numpy as np
from Data import *
from graphviz import Digraph


debug_tree = "true"
class TreeGraph:
    def __init__(self, name, tree):
        self.name = name
        self.e = Digraph('name', filename='Tree.gv')
        self.e.attr(size='8,5')
        self.column_list = tree.GetColumnList()
        return
        
    def AddEntry(self, depth, typeofentry, column_obj1, column_obj2, decision, result):
        box1_name = column_obj1.GetName() + str(depth)
        if (typeofentry == "mid-node"):
            self.e.attr('node', shape='box')
            box2_name = column_obj2.GetName() + str(depth)
            self.e.node(box1_name, label = column_obj1.GetName())
            self.e.node(box2_name, label = column_obj2.GetName())

            self.e.edge(box1_name, box2_name, decision)
        else:
            self.e.attr('node', shape='circle')
            self.e.edge(box1_name, result, decision)

    def View(self):
        self.e.view()
        return        
        
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
        
    def GetFirstCol(self, type_of_col):
        for i in self.column_list:
            if (i.GetTypeOfColumn() == type_of_col):
                return i
            elif (i.GetTypeOfColumn() == type_of_col):
                return i
        return "null_obj"
            
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
    

    def GetOutputColumnEntropy(self):
        output_column = self.column_list[self.output_column_num]
        #print("GetOutputColumnEntropy: ",output_column.GetColumnEntropy(), " ", output_column.GetColumnData())
        return output_column.GetColumnEntropy()
    
    def GetOutputColumnAnswer(self):
        output_column = self.column_list[self.output_column_num]
        out =  output_column.GetUserQuestion(0)
        return out
    
    def GetColumnList(self):
        return self.column_list


def PrintLevel(level):
    for i in np.arange(level):
        print(".", end = " ")   
        
            
def GetSubTree(depth, original_tree, remove_column_index, decision_value, query):
    #print("GetSubTree: remove_column_index: ", remove_column_index, "Decision: ", decision_value," depth:", depth)
    #print("GetSubTree:\n")
    t = Tree(depth)
    column_obj = original_tree.GetColumn(remove_column_index)
    index_values = column_obj.GetIndexValues(decision_value)
    #print("GetSubtree: Removing column: ", column_obj.GetName())
    for i in np.arange(original_tree.GetTotalColumn()):
        if (i != remove_column_index):
            valid_column = original_tree.GetColumn(i)
            coltype, name, attr1, attr2, ncolumn = valid_column.GetFilterColumn(index_values)

            c1 = ClassColumn(coltype, name, attr1, attr1, ncolumn)
            #print("GetSubTree: ", name, " .. ", ncolumn)

            t.AddColumn(c1)
    
    e = t.GetOutputColumnEntropy()
    #print("GetSubtree: Got Entropy: ", e)
    #print("GetSubTree:...Total columns in new tree:", t.GetTotalNumInputColumn())
    
    # Entropy of 0 is leaf
    if (e == 0):
        question = "Depth: "  +  str(depth) +  query + t.GetOutputColumnAnswer()
        PrintLevel(depth)
        print(question)
        return t, "leaf", e, question
    
    # Only one column left- just print probabiilty
    if (t.GetTotalNumInputColumn() <= 1):
        input_column =  t.GetFirstCol("input")
        output_column = t.GetFirstCol("output")

        in_p,in_unique_val = input_column.GetProbability()
        question = ""
        #print("..vyas", in_unique_val)
        for i in np.arange(len(in_unique_val)):
            in_attr_value = in_unique_val[i]
            #print(in_attr_value)
            index_values = input_column.GetIndexValues(in_attr_value)
            #print(index_values)
            coltype, name, attr1, attr2, ncolumn = output_column.GetFilterColumn(index_values)
            nc = ClassColumn(coltype, name, attr1, attr2, ncolumn)
            nc_p, nc_unique_val = nc.GetProbability()
            
            for j in np.arange(len(nc_p)):
                question = "Depth: " + str(depth) +  input_column.GetUserQuestion(i) + nc.GetUserQuestion(j) + " with P = " + str(nc_p[j])
                PrintLevel(depth)
                print(question, "\n")                
                #print("GetSubTree:....", input_column.GetUserQuestions(), i, j, input_column.GetUserQuestion(i) )
        return t, "leaf" , e, question
    
    #print("GetSubTree: midnode")
    PrintLevel(depth)
    question = "Depth: "  +  str(depth) + query
    print(question)
    return t, "mid-node", e, " "
      

def PreOrderTraversal(graph, t, valid_flag, depth, query):
    #print("PreOrderTraversal: ..depth..", depth)

    #t.Print()
    t.ComputeEntropy()
    #PrintLevel(depth)

    if (valid_flag == "valid"):
        col_num = t.GetMinEntropyNode()
        column_obj = t.GetColumn(col_num)

        unique_decision_count = column_obj.GetNumUniqueValueCount()
        #print("PreOrderTraversal: unique_decision_count on ", unique_decision_count)

        for i in np.arange(unique_decision_count):
            #print("hoy..", column_obj.GetName(),  i, column_obj.GetUniqueValues() )
            filter_data = column_obj.GetUniqueData(i)
            str_var = " Is " + column_obj.GetName() + " " + filter_data +  " ? "

            new_tree, tree_type, e, q = GetSubTree(depth+1, t, col_num, filter_data, str_var)
            if (tree_type == "mid-node"):
                #print("PreOrderTraversal: Filtering on i: ", i, "f : ",filter_data)
                #PrintLevel(depth)
                #q = query + str_var
                PreOrderTraversal(graph, new_tree, "valid", depth+1, q)
            else:
                #print(q)
                #str_var = " Is " + column_obj.GetName() + " " + filter_data +  " ? "
                #print(str_var , end " ")

                continue
    return            

def TestGraph(t):
    g = TreeGraph("play", t)
    c = g.column_list
    g.AddEntry(0, "mid-node", c[0], c[1], "testing1", "yes")
    g.AddEntry(0,"leaf", c[0], "", "test2", "no")
    g.View()

            
def Test():
    i1= [ "rainy", "rainy", "overcast", "sunny", "sunny", "sunny", "overcast", "rainy", "rainy", "sunny", "rainy", "overcast", "overcast", "sunny", "rainy"]
    i2 = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot","mild",  "hot" ]
    i3 = ["high", "high", "high", "high", "normal", "normal", "normal", "high","normal", "normal", "normal", "high", "normal", "high", "high", ]
    i4 = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true","false", "true", "true" ]
    o1 = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
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
    g = TreeGraph("play", t)

    PreOrderTraversal(g, t,"valid", 0,"")
    #c2.PiePlot()
    TestGraph(t)
    
    
    return

if __name__ == "__main__":
    # execute only if run as a script
    Test()
