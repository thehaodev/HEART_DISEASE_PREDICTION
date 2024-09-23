import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean)
df['ca'] = df.ca.fillna(df.ca.mean())

# Distribution of target vs age
sns.set_context('paper', font_scale=1, rc={'font.size': 3, 'axes.titlesize': 15, 'axes.labelsize': 10})
ax = sns.catplot(kind='count', data=df, x='age', hue='target', order=df['age'].sort_values().unique())
ax.ax.set_xticks(np.arange(0, 88, 5))

# barplot of age vs sex
sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')
plt.title('Distrubution of age vs sex thi the target class')

plt.title('Variation of Age for each target class')
plt.show()
