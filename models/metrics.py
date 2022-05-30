'''
This file defines the analysis of the predictions generated
by each of the models.
'''
from skelarn.metrics import ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt

def confusion_matrix(y_labels, predictions, title):
    '''
    Plots the confusion matrix given the labels and the predictions on the testing set.
    
    Params:
        - 'y_labels': the true labels of each system
        - 'predictions': the predictions generated after evaluating on the test set
        - 'title': the title for the confusion matrix and save location
    '''
    
    cm = confusion_matrix(y_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm).plot()
    disp.ax_.set_title(title)
    disp.figure_.savefig('../figures/cm_' + title + '.png', bbox_inches = 'tight')

def precision_recall_tradeoff_curve(y_labels, predictions, title):
    '''
    Plots the precision, recall, and F1 score for various decision thresholds.
    Params:
        - 'y_labels': the true labels of each system
        - 'predictions': the predictions generated after evaluation on test/val set
        - 'title': the title for the plot and save location
    '''
    
    precision, recall, thresholds = precision_recall_curve(y_labels, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    plt.plot(thresholds, precision[:-1], label = 'Precision')
    plt.plot(thresholds, recall[:-1], label = 'Recall')
    plt.plot(thresholds, f1_scores[:-1], label = 'F1')
    
    # Plot threshold for f1 score
    plt.axhline(y=0.7, linestyle = '--', c='g', label = 'F1=0.7 Threshold')
    plt.title('Precision, Recall, F1 Score Tradeoff, ' + title)
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.legend()
    
    plt.savefig('../figures/' + title + 'metrics_plot.png')