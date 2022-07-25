# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:19:00 2022

@author: nurul
"""

import numpy as np
import seaborn as sns
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
#%%


class cramax:
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher,
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
class EDA:
    def visualization(self,con_col,cat_col,df):
        for con in con_col:
            plt.figure()
            sns.distplot(df[con],color='red')
            plt.show()
        for cat in cat_col:
            plt.figure()
            sns.countplot(df[cat],color='yellow')
            plt.show()
    def countplot_graph(self,cat_col,df):
        for i in cat_col:
            plt.figure()
            sns.countplot(df[i],hue=df['output'],color='green')
            plt.show()   