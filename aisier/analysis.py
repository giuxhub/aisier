import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class Analysis:
    def __init__(self, path):
        self.path = os.path.abspath(os.path.join(path, 'analysis'))

        self.dataset = None
        self.X = None
        self.X_norm = None
        self.Y = None
        self.attributes = None
        self.attributes_without_target = None
        self.labels = None
        self.num_features = 0

    def _chi_squared(self):
        chi_selector = SelectKBest(chi2, k=self.num_features)
        chi_selector.fit(self.X_norm, self.Y.ravel())
        chi_support = chi_selector.get_support()

        return chi_support

    def _rfe_feature(self):
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=self.num_features, step=10, verbose=5)
        rfe_selector.fit(self.X_norm, self.Y.ravel())
        rfe_support = rfe_selector.get_support()

        return rfe_support

    def _lasso_features(self):
        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=self.num_features)
        embeded_lr_selector.fit(self.X_norm, self.Y.ravel())
        embeded_lr_support = embeded_lr_selector.get_support()

        return embeded_lr_support

    def _random_forest_features(self):
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=self.num_features)
        embeded_rf_selector.fit(self.X_norm, self.Y.ravel())
        embeded_rf_support = embeded_rf_selector.get_support()

        return embeded_rf_support

    # https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    def _correlation_matrix(self):
        dataset = np.concatenate((self.dataset.train,
                                  self.dataset.test,
                                  self.dataset.validation))

        df = pd.DataFrame(dataset, columns=self.attributes)
        corr = df.corr(method='pearson')

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

        ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')

        fig.savefig(os.path.join(self.path, 'corr_matrix.pdf'))

    # Thanks to https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    def _pca_projection(self, n_components):
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(self.X)

        print('dataset shape {}'.format(self.X.shape))
        print('projected shape {}'.format(projected.shape))

        plt.figure('PCA Projection')
        plt.title('PCA Projection')

        plt.scatter(projected[:, 0], projected[:, 1],
                    c=self.Y, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('Spectral', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()

        plt.savefig(os.path.join(self.path, 'pca_projection.pdf'))

    def _pca_explained_ratio(self):
        pca = PCA()
        pca.fit_transform(self.X)

        plt.figure('PCA explained ratio')
        plt.title('PCA explained ratio')
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')

        plt.savefig(os.path.join(self.path, 'pca_explained_ratio.pdf'))

    def analyze_features(self, dataset, num_features, attributes):
        if num_features > len(attributes):
            return 'num_features cannot be greater that the total number of features in the dataset'

        self.num_features = num_features
        self.attributes = np.array(attributes)
        self.attributes_without_target = np.array(attributes[:-1])
        self.dataset = dataset
        self.X = np.concatenate((dataset.X_train,
                                 dataset.X_test,
                                 dataset.X_val))

        self.Y = np.concatenate((dataset.train[:, -1:],  # here we need the non-to_categorical version of y
                                 dataset.test[:, -1:],
                                 dataset.validation[:, -1:]))

        self.X_norm = MinMaxScaler().fit_transform(self.X)

        feature_selection_df = pd.DataFrame({'Feature': self.attributes_without_target,
                                             'Chi-2': self._chi_squared(),
                                             'RFE': self._rfe_feature(),
                                             'Logistics': self._lasso_features(),
                                             'Random Forest': self._random_forest_features()})

        # count the selected times for each feature
        feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
        # display the top 100
        feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
        feature_selection_df.index = range(1, len(feature_selection_df) + 1)
        feature_selection_df.head(self.num_features)

        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=feature_selection_df.values, colLabels=feature_selection_df.columns, loc='center')

        fig.tight_layout()

        plt.savefig(os.path.join(self.path, 'feature_table.pdf'))
        plt.clf()

        self._correlation_matrix()
        self._pca_explained_ratio()
        self._pca_projection(2)

        return None
