import numpy as np
import pandas as pd


class FeatureSelector:
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = features
        self.feature_names = self.features.keys().to_list()

        self.features_selected = None
        self.feature_names_selected = None
        self.features_removed = None
        self.feature_names_removed = None

        self.labels = labels
        self.labels_name = self.labels.keys().to_list()
        self.ranked = None

    def select_by_rfe(self, estimator=RandomForestGressor(), n_features_to_select=10):
        """
        通过递归特征消除法选择特征
        :param estimator: 估计器
        :param n_features_to_select: 选择的特征数量
        :return:
        """
        from sklearn.feature_selection import RFE
        from sklearn.multioutput import MultiOutputRegressor
        from tqdm import tqdm

        # use MultiOutputRegressor to support multi-output regression
        multioutput_estimator = MultiOutputRegressor(estimator)
        # construct and fit MultiOutputRegressor
        multioutput_estimator.fit(self.features, self.labels)
        # init empty list to store rfe results
        rfes = []
        # loop through each feature
        for i, estimator in tqdm(enumerate(multioutput_estimator.estimators_)):
            # construct RFE
            rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
            # fit RFE
            rfe.fit(self.features, self.labels.iloc[:, i])
            # append rfe results to list
            rfes.append(rfe.ranking_)
        # get the min ranking of each feature
        rankings = np.min([rfe.ranking_ for rfe in rfes], axis=0)
        # select the features with the min ranking
        selected_features_mask = (
            rankings <= n_features_to_select if n_features_to_select else rankings == 1
        )
        # get selected features and their names by mask
        self.feature_names_selected = [
            feature
            for feature, mask in zip(self.feature_names, selected_features_mask)
            if mask
        ]
        self.features_selected = self.features[self.feature_names_selected]
        # get removed features and their names by mask
        self.feature_names_removed = [
            feature
            for feature, mask in zip(self.feature_names, selected_features_mask)
            if not mask
        ]
        self.features_removed = self.features[self.feature_names_removed]

        # get the ranking table
        self.ranked = (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "ranking": rankings,
                }
            )
            .sort_values(by="ranking")
            .reindex(drop=True)
        )
        return self

    def select_by_rfecv(self, estimator=RandomForestGressor(), cv=5, scoring=None):
        from sklearn.feature_selection import RFECV
        from tqdm import tqdm

        rfecvs = []
        all_support_masks = []
        for i in tqdm(range(self.features.shape[1])):
            rfecv = RFECV(estimator=estimator, cv=cv, scoring=scoring)
            rfecv.fit(self.features, self.labels.iloc[:, i])
            rfecvs.append(rfecv)
            all_support_masks.append(rfecv.support_)

        # get the union of all support masks by any operation
        union_support_mask = np.any(all_support_masks, axis=0)

        # self.feature_names_removed = [
        #     feature for feature, mask in zip(self.features_name, ~union_support_mask) if mask
        # ]

        # self.feature_names_selected = [
        #     feature for feature, mask in zip(self.features_name, union_support_mask) if mask
        # ]
        self.feature_names_selected = self.feature_names[union_support_mask].to_list()
        self.feature_names_removed = self.feature_names[~union_support_mask].to_list()
        self.features_removed = self.features[self.feature_names_removed]
        self.features_selected = self.features[self.feature_names_selected]

        ranking = pd.DataFrame({"feature": self.feature_names})
        ranking_target = []
        for i, rfecv in enumerate(rfecvs):
            ranking_target.append(f"ranking_target_{i+1}")
            ranking[f"ranking_target_{i+1}"] = rfecv.ranking_
        ranking["syn_ranking"] = ranking[ranking_target].apply(
            lambda row: row.min(), axis=1
        )
        self.ranked = ranking.sort_values(by="syn_ranking").reindex(drop=True)
        return self

    def select_by_lasso(self, cv=5, alpha=0.1):
        from sklearn.linear_model import LassoCV
        from sklearn.multioutput import MultiOutputRegressor

        # use MultiOutputRegressor to support multi-output regression
        multioutput_estimator = MultiOutputRegressor(LassoCV(cv=cv, alpha=alpha))
        # construct and fit MultiOutputRegressor
        multioutput_estimator.fit(self.features, self.labels)

        selected_features_mask = np.any(
            [estimator.coef_ != 0 for estimator in multioutput_estimator.estimators_],
            axis=0,
        )
        self.feature_names_selected = self.feature_names[
            selected_features_mask
        ].to_list()
        self.features_selected = self.features[self.feature_names_selected]
        self.feature_names_removed = self.feature_names[
            ~selected_features_mask
        ].to_list()
        self.features_removed = self.features[self.feature_names_removed]
        return self

    def select_by_importance(self, estimator=RandomForestRegressor(), top_n=10):
        from sklearn.multioutput import MultiOutputRegressor

        multioutput_rf = MultiOutputRegressor(estimator)
        multioutput_rf.fit(self.features, self.labels)
        # get sum feature importances
        feature_importances = np.sum(
            [
                estimator.feature_importances_
                for estimator in multioutput_rf.estimators_
            ],
            axis=0,
        )
        importances = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": feature_importances,
            }
        ).sort_values(by="importance", ascending=False)
        importances = importances.reset_index(drop=True)

        if top_n and top_n <= len(importances):
            self.feature_names_selected = importances["feature"].head(top_n).to_list()
        else:
            self.feature_names_selected = importances[importances["importance"] > 0][
                "feature"
            ].to_list()

        self.features_selected = self.features[self.feature_names_selected]
        self.feature_names_removed = importances[
            ~importances["feature"].isin(self.feature_names_selected)
        ]["feature"].to_list()
        self.features_removed = self.features[self.feature_names_removed]
        return self

    def select_by_correlation(self, threshold=0.5):
        self.correlation_matrix = self.features.corr()
        # upper triangle of the correlation matrix
        upper_triangle = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )

        # select features with correlation greater than threshold
        self.feature_names_selected = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]
        self.features_selected = self.features[self.feature_names_selected]
        self.feature_names_removed = [
            column
            for column in self.features.columns
            if column not in self.feature_names_selected
        ]
        self.features_removed = self.features[self.feature_names_removed]
        return self

    def select_by_vif(self, threshold=5.0):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import statsmodels.api as sm

        # add a constant to the features
        features_with_constant = sm.add_constant(self.features)
        # calculate the variance inflation factor for each feature
        vif = pd.DataFrame()
        vif["feature"] = self.feature_names
        vif["vif"] = [
            variance_inflation_factor(features_with_constant.values, i + 1)
            for i in range(len(self.feature_names))
        ]
        # select the features with a vif below the threshold
        self.feature_names_selected = vif[vif["vif"] < threshold]["feature"].tolist()
        self.features_selected = self.features[self.feature_names_selected]
        self.feature_names_removed = [
            feature_name
            for feature_name in self.feature_names
            if feature_name not in self.feature_names_selected
        ]
        self.features_removed = self.features[self.feature_names_removed]
        return self

    def apply_pca(self, explained_variance_ratio_threshold=0.95):
        from sklearn.decomposition import PCA

        # apply PCA to the selected features
        pca = PCA(n_components=explained_variance_ratio_threshold)
        self.features_selected = pca.fit_transform(self.features_selected)
        # get correspond of main component and original feature
        self.components_df = pd.DataFrame(pca.components_, columns=self.feature_names)
        # construct new list of feature names
        self.feature_names_selected = [
            f"PC_{i+1}" for i in range(self.features_selected.shape[1])
        ]
        return self
