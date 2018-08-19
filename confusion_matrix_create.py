import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

    


print_confusion_matrix(matrix1,GENRES, (7,7),14)
print_confusion_matrix(matrix2,GENRES, (7,7),14)
print_confusion_matrix(matrix3,GENRES, (7,7),14)
print_confusion_matrix(matrix4,GENRES, (7,7),14)
print_confusion_matrix(matrix5,GENRES, (7,7),14)
print_confusion_matrix(matrix6,GENRES, (7,7),14)
print_confusion_matrix(matrix7,GENRES, (7,7),14)
print_confusion_matrix(matrix8,GENRES, (7,7),14)




import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')



energy1 = [2700, 420, 59]
energy2 = [3300,501,73]
energy3 = [4140,654,94]
energy4 = [3120,602,72]
energy5 = [4500,666,104]
energy6 = [1800,297,40]
energy7 = [2520,406,57]
energy8 = [4440,823,100]





x = ['830M', '1060 GTX', 'K80']

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy8, color='green')
plt.xlabel("GPU used for training")
plt.ylabel("Time in seconds")
plt.title("Training time of the architecure on Hardwares used")

plt.xticks(x_pos, x)

plt.show()