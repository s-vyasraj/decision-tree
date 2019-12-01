# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:59:14 2019

@author: vyas
"""
import numpy as np
from  Data import *
from  Tree import *

import csv
import codecs
  
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
    c4 = ClassColumn("input", "Urine pain", "yes", "no", ndata[:,3])
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



                
    
def Dtree(graph,t):
    print(__name__)
    PreOrderTraversal(graph, t,"valid", 0,"")
          
        
def main():
    file_name = "G:/code/decision-tree/data/nephra/diagnosis.clean.data"
    tree, ndata = ImportData(file_name)
    g = TreeGraph("play", tree)

    Dtree(g, tree)


if __name__ == "__main__":
    # execute only if run as a script
    main()