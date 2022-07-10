from collections import defaultdict
import matplotlib.pyplot as plt
from numpy import linalg as la
import networkx as nx
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd


class ValuesMatrix:

    def __init__(self,n,m) -> None:
        self.valuesMatrix = defaultdict(list) 
        self.longTerm =  np.zeros((n,m))
        self.edges = list()

 
        
    def importData(self, df):
        '''
        df = .csv
        '''
        self.df = pd.read_csv(df)
    
    def makeGraph(self,edges):
        G = nx.MultiGraph()
        G.add_edges_from(edges)
        self.G = G
        nx.draw(G, with_labels=True)



