import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Accuracy_CNN_values.txt', sep='-', header=0)

plt.figure(figsize=(14,6))
plt.title("CNN Model Accuracy")
sns.lineplot(x = 'Epoch ', y = ' Accuracy ', data=data, label='Train')
sns.lineplot(data=data[' Val_accuracy'], label='Test')