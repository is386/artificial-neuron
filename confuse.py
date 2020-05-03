import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# Confusion Matrix for Logistic Activation
array = [[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2],
         [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0],
         [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]


df_cm = pd.DataFrame(array, index=[i for i in range(2, 16)],
                     columns=[i for i in range(2, 16)])
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.2)
sn.heatmap(df_cm, annot=True)
plt.show()
