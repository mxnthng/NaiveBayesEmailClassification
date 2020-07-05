import re # module bieu thuc chinh quy regular expression
import pandas as pd # thu vien xu li du lieu
import matplotlib.pyplot as plt # thu vien truc quan hoa du lieu

dataset = pd.read_csv("dataset3000.csv") # doc file dataset
dataset.drop_duplicates(inplace = True) # xoa trung lap trong dataset

dataset["spam"].value_counts().plot(kind='bar') # ve bieu do
plt.show() # show bieu do