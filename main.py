import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

df = pd.read_csv(r'data\all2.csv', index_col=0)
df.columns = list(map(lambda x: x[:8], df.columns))
