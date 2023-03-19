import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("Training.csv")

df.drop('Unnamed: 133', axis=1, inplace=True)
x = df.drop('prognosis', axis = 1)
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier()

tree.fit(x_train, y_train)

pickle.dump(tree,open('model1.pkl','wb'))
model = pickle.load(open('model1.pkl','rb'))

#pred = tree.predict([[0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,1,0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0]])
#print(pred)