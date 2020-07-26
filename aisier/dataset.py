import os
import numpy as np
import pandas as pd
from multiprocessing import Process, Manager
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Dataset:
    def __init__(self, path):
        self.path = os.path.abspath(os.path.join(path, 'dataset'))
        self.train_path = os.path.join(self.path, 'data_train.csv')
        self.test_path = os.path.join(self.path, 'data_test.csv')
        self.validation_path = os.path.join(self.path, 'data_validation.csv')
        self._loaded = False
        self.n_labels = 0
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.X_val = None
        self.Y_val = None
        self.X = None
        self.Y = None
        self.train = None
        self.test = None
        self.validation = None

    def loader(self, queue, pathname):
        data = dict()

        if not os.path.exists(pathname):
            data[pathname] = None
        else:
            data[pathname] = pd.read_csv(os.path.join(self.path, pathname), sep=',', header=None).to_numpy()

        queue.put(data)

    def saver(self, pathname, data):
        data.to_csv(os.path.join(self.path, pathname), sep=',', mode='a', header=None, index=None, chunksize=512)

    def _save(self, train_data, test_data, validation_data):
        processes = [
            Process(target=self.saver, args=(self.train_path, train_data,)),
            Process(target=self.saver, args=(self.test_path, test_data,)),
            Process(target=self.saver, args=(self.validation_path, validation_data,))
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def _load(self, which=None):
        output = Manager().Queue()
        data = dict()
        processes = list()

        if which == 'train':
            processes.append(Process(target=self.loader, args=(output, self.train_path,)))
        elif which == 'test':
            processes.append(Process(target=self.loader, args=(output, self.test_path,)))
        elif which == 'validation':
            processes.append(Process(target=self.loader, args=(output, self.validation_path,)))
        else:
            processes.append(Process(target=self.loader, args=(output, self.train_path,)))
            processes.append(Process(target=self.loader, args=(output, self.validation_path,)))
            processes.append(Process(target=self.loader, args=(output, self.test_path,)))

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        results = [output.get() for _ in range(output.qsize())]
        for res in results:
            for key, value in res.items():
                if value is None:
                    return 'error loading dataset'

                data[key] = value

        if which == 'train':
            self.train = data[self.train_path]
            u = self.train[:, -1:]
        elif which == 'test':
            self.test = data[self.test_path]
            u = self.test[:, -1:],
        elif which == 'validation':
            self.validation = data[self.validation_path]
            u = self.validation[:, -1:]
        else:
            self.train = data[self.train_path]
            self.test = data[self.test_path]
            self.validation = data[self.validation_path]

            u = np.concatenate((self.train[:, -1:],
                                self.test[:, -1:],
                                self.validation[:, -1:]))

        self.n_labels = len(np.unique(u))

        return None

    def is_loaded(self):
        return self._loaded

    def load_dataset(self):
        err = self._load()
        if err is not None:
            return err

        self.X_train, self.Y_train = self.train[:, :-1], tf.keras.utils.to_categorical(self.train[:, -1:], self.n_labels)
        self.X_test, self.Y_test = self.test[:, :-1], tf.keras.utils.to_categorical(self.test[:, -1:], self.n_labels)
        self.X_val, self.Y_val = self.validation[:, :-1], tf.keras.utils.to_categorical(self.validation[:, -1:], self.n_labels)

        self._loaded = True

        return None

    def load_test(self):
        if self.is_loaded():
            return None

        if not self.has_test():
            return '{} does not exist'.format(self.test_path)
        else:
            err = self._load(which='test')
            if err is not None:
                return err

            self.X_test, self.Y_test = self.test[:, :-1], tf.keras.utils.to_categorical(self.test[:, -1:], self.n_labels)

        return None

    def build_dataset(self, path, test, validation):
        dataset_path = os.path.join(self.path, path)
        if not os.path.exists(dataset_path):
            return '{} does not exist in {}'.format(path, self.path)

        data = pd.read_csv(os.path.join(dataset_path), sep=',', header=None).to_numpy()

        self.X = data[:, :-1]
        self.Y = data[:, -1:]

        self.n_labels = len(np.unique(self.Y))

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train,
                                                                              test_size=validation)

        self.train = np.hstack((self.X_train, self.Y_train))
        self.test = np.hstack((self.X_test, self.Y_test))
        self.validation = np.hstack((self.X_val, self.Y_val))

        self.Y_train = tf.keras.utils.to_categorical(self.Y_train, self.n_labels)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test, self.n_labels)
        self.Y_val = tf.keras.utils.to_categorical(self.Y_val, self.n_labels)

        self._save(pd.DataFrame(self.train),
                   pd.DataFrame(self.test),
                   pd.DataFrame(self.validation))

        self._loaded = True

        return None

    def optimize(self, path):
        dataset_path = os.path.join(self.path, path)

        if not os.path.exists(dataset_path):
            return '{} does not exist in {}'.format(path, self.path)

        data = pd.read_csv(dataset_path, sep=',', header=None)
        n_tot = len(data)
        print('loaded {} total samples'.format(n_tot))

        unique = data.drop_duplicates()
        n_uniq = len(unique)
        print('found {} unique samples'.format(n_uniq))

        self.saver('dataset_unique.csv', unique)

        return None

    def has_train(self):
        return os.path.exists(self.train_path)

    def has_test(self):
        return os.path.exists(self.test_path)

    def has_validation(self):
        return os.path.exists(self.validation_path)

    def exists(self):
        return self.has_train() and \
               self.has_test() and \
               self.has_validation()
