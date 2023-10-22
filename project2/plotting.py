import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 

sns.set_theme()

df_gd = pd.read_csv('OLS-GD.csv')
df_sgd = pd.read_csv('OLS-SGD.csv')

sns.heatmap(data=df_sgd.corr(), annot=True)
# plt.xscale('log')
# plt.yscale('log')
plt.tight_layout()
plt.show()