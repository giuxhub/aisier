import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from importlib.machinery import SourceFileLoader

from aisier.analysis import Analysis
from aisier.dataset import Dataset
from aisier.plot import Plot


class Aisier:
    def __init__(self, path):
        # base info
        self.path = os.path.abspath(path)
        # model related data
        self.model = None
        self.accu = None
        self.model_with_weights_path = os.path.join(self.path, 'model.h5')
        # training related data
        self.dataset = Dataset(self.path)
        self.label_path = os.path.join(self.path, 'labels.json')
        self.labels = None
        self.history_path = os.path.join(self.path, 'history.json')
        self.history = None
        self.attributes_path = os.path.join(self.path, 'attributes.names')
        self.attributes = None
        # plotting related data
        self.plotter = Plot(self.path)
        # feature selection related data
        self.analyzer = Analysis(self.path)

    def _load_function(self, module_name, function_name):
        module_path = os.path.join(self.path, module_name)
        symbols = SourceFileLoader("", module_path).load_module()
        if function_name not in symbols.__dict__:
            return '{} does not define a {} function'.format(module_path, function_name)

        return symbols.__dict__[function_name]

    def _save_model(self):
        print('updating {}...'.format(self.model_with_weights_path))
        self.model.save(self.model_with_weights_path)

    def _save_history(self, history):
        print('saving model history in {}...'.format(self.history_path))
        with open(self.history_path, 'w') as fp:
            json.dump(history, fp)

    def _load_attributes(self):
        if not os.path.exists(self.attributes_path):
            return '{} does not exist'.format(self.attributes_path)

        with open(self.attributes_path, 'r') as fp:
            self.attributes = [x.strip('\n') for x in fp.readlines()]

        return None

    def _load_labels(self):
        if self.labels is not None:
            return None

        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as fp:
                try:
                    self.labels = json.load(fp)
                except json.JSONDecodeError as err:
                    return 'error {} while decoding {}'.format(err, self.history_path)
        return None

    def _is_dataset_loaded(self):
        return self.dataset.is_loaded()

    def _load_dataset(self):
        if not self._is_dataset_loaded():
            if self.dataset.exists():
                self.dataset.load_dataset()
            else:
                return 'dataset does not exist'

        return None

    def _load_trained_model(self):
        print('loading model from {}...'.format(self.model_with_weights_path))
        self.model = tf.keras.models.load_model(self.model_with_weights_path)

    def load(self):
        print('loading project {}...'.format(self.path))

        if not os.path.exists(self.path):
            return '{} does not exist'.format(self.path)

        # If the model was already trained...
        if os.path.exists(self.model_with_weights_path):
            self._load_trained_model()
        else:
            # Build the model loading the one written into model.py
            model_builder = self._load_function('model.py', 'build_model')
            self.model = model_builder(True)

        return None

    def train(self):
        if self.model is None:
            model_builder = self._load_function('model.py', 'build_model')
            self.model = model_builder(True)

        train_model = self._load_function('model.py', 'train_model')
        self.history = train_model(self.model, self.dataset).history

        # save model training history
        self._save_history(self.history)

        # save model structure and weights
        self._save_model()

    def prepare(self, filename, label):
        prepare_dataset = self._load_function('prepare.py', 'prepare_dataset')
        data = prepare_dataset(filename, label)
        self.dataset.saver('dataset.csv', data)

    def build_dataset(self, test, validation):
        return self.dataset.build_dataset('dataset.csv', test, validation)

    def optimize(self, filename):
        return self.dataset.optimize(filename)

    def plot_history(self):
        if not os.path.exists(self.history_path):
            return 'file {] does not exist'.format(self.history_path)

        with open(self.history_path, 'r') as fp:
            try:
                self.history = json.load(fp)
                self.plotter.plot_history(self.history)
            except json.JSONDecodeError as err:
                return 'error {} while decoding {}'.format(err, self.history_path)

        return None

    def plot_roc(self):
        err = self._load_dataset()
        if err is not None:
            return err

        if self.model is None:
            if os.path.exists(self.model_with_weights_path):
                self._load_trained_model()
            else:
                return 'model {} does not exist.'.format(self.model_with_weights_path)

        self.plotter.plot_roc(self.model, self.dataset)

        return None

    def plot_confusion_matrix(self):
        err = self.dataset.load_test()
        if err is not None:
            return err

        err = self._load_labels()
        if err is not None:
            return err

        if self.model is None:
            if os.path.exists(self.model_with_weights_path):
                self._load_trained_model()
            else:
                return 'model {} does not exist.'.format(self.model_with_weights_path)

        Y_pred = np.argmax(self.model.predict(self.dataset.X_test), axis=1)
        cm = confusion_matrix(np.argmax(self.dataset.Y_test, axis=1), Y_pred)

        self.plotter.plot_confusion_matrix(cm, self.labels)

        return None

    def view(self):
        self.plot_history()
        self.plot_confusion_matrix()
        self.plot_roc()

    def analyze(self, num_features):
        err = self._load_dataset()
        if err is not None:
            return err

        err = self._load_attributes()
        if err is not None:
            return err

        self.analyzer.analyze_features(self.dataset, num_features, self.attributes)

