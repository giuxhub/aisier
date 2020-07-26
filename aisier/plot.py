import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Plot:
    def __init__(self, path):
        self.path = os.path.abspath(os.path.join(path, 'plot'))

    def plot_history(self, history):
        # summarize history for accuracy
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.path, 'history_accuracy.pdf'))

        # clean plot
        plt.clf()

        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.path, 'history_loss.pdf'))

        return None

    def plot_roc(self, model, dataset):
        print('computing ROC curve on {} samples...'.format(len(dataset.X_test)))
        y_pred = model.predict(dataset.X_test)
        fpr, tpr, thresholds = roc_curve(dataset.Y_test.ravel(), y_pred.ravel())

        plt.figure('ROC Curve')
        plt.title('ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()

        plt.savefig(os.path.join(self.path, 'roc.pdf'))

    def plot_confusion_matrix(self, cm, target_names, normalize=True, fontsize=18):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        cmap = plt.get_cmap('Spectral')

        plt.figure('Confusion Matrix', figsize=(10, 10))
        plt.title('Confusion Matrix', fontsize=fontsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45, fontsize=fontsize)
            plt.yticks(tick_marks, target_names, fontsize=fontsize)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, '{:0.2f}'.format(cm[i, j]),
                         horizontalalignment='center',
                         color='white' if cm[i, j] > thresh else 'black',
                         fontsize=fontsize)
            else:
                plt.text(j, i, '{:,}'.format(cm[i, j]),
                         horizontalalignment='center',
                         color='white' if cm[i, j] > thresh else 'black',
                         fontsize=fontsize)

        plt.ylabel('Predicted label')
        plt.xlabel('True label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
        plt.savefig(os.path.join(self.path, 'confusion_matrix.pdf'))
