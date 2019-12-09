from pathlib import Path

import category_encoders as ce
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

dataset_path = Path("resources")
train_path = dataset_path / "train_data.csv"
test_path = dataset_path / "test_data.csv"


def submit(predictions):
    pd.DataFrame(
        {"id": range(len(predictions)), "y": np.expm1(predictions)}).to_csv(
        "submission.csv", index=False)


def lgb_cv(_train, _test, _target, model_params, train_params, cat_idx,
           fold_schema):
    oof = np.zeros(len(_train))
    predictions = np.zeros(len(_test))

    for i, (trn_idx, val_idx) in enumerate(fold_schema.split(_train)):
        print("fold: {}".format(i))
        trn_data = lgb.Dataset(_train.iloc[trn_idx],
                               label=_target.iloc[trn_idx])
        val_data = lgb.Dataset(_train.iloc[val_idx],
                               label=_target.iloc[val_idx])
        model = lgb.train(model_params,
                          trn_data,
                          valid_names=['train', 'valid'],
                          valid_sets=[trn_data, val_data],
                          # categorical_feature=cat_idx,
                          **train_params)
        oof[val_idx] = model.predict(_train.iloc[val_idx],
                                     num_iteration=model.best_iteration)
        print(mean_absolute_error(np.expm1(_target.iloc[val_idx]),
                                  np.expm1(oof[val_idx])))
        predictions += model.predict(_test,
                                     num_iteration=model.best_iteration) / fold_schema.n_splits
    print(mean_absolute_error(np.expm1(_target), np.expm1(oof)))

    return predictions


def main():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target_col = "salary"
    target = train[target_col]
    target = target.map(np.log1p)
    train.drop(columns=[target_col], inplace=True)
    train["commute_multi_position"] = train["commute"] * train["position"]
    test["commute_multi_position"] = test["commute"] * test["position"]
    train["age_multi_position"] = train["age"] * train["position"]
    test["age_multi_position"] = test["age"] * test["position"]
    train["age_multi_commute"] = train["age"] * train["commute"]
    test["age_multi_commute"] = test["age"] * test["commute"]
    train["is_test"] = 0
    test["is_test"] = 1
    data = pd.concat([train, test])
    #
    # data['city'] = data['area'].apply(
    #     lambda x: 1 if '東京' in x or '大阪' in x else 0)

    # area_encode = LabelEncoder()
    # area_encode.fit(
    #     list(set(train["area"].unique()) & set(test["area"].unique())))
    # train["area"] = area_encode.transform(train["area"])
    # test["area"] = area_encode.transform(test["area"])
    area_encoder = LabelEncoder()
    area_encoder.fit(data["area"])
    data["area"] = area_encoder.transform(data["area"])
    #
    data[data['age'] >= 60]['age'] = 60

    # id 8117

    n_splits = 4
    kf = KFold(n_splits=n_splits, random_state=71, shuffle=True)
    model_params = {
        'boosting_type': 'gbdt',
        'objective': 'fair',
        'metric': 'fair',
        'n_estimators': 20000,
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'subsample': 0.7
    }

    train_params = {
        'early_stopping_rounds': 200,
        'verbose_eval': -1
    }

    # id is wasted col
    drop_cols = ["id", "service_length"]
    data.drop(columns=drop_cols, inplace=True)
    # train.drop(columns=drop_cols, inplace=True)
    # test.drop(columns=drop_cols, inplace=True)

    cat_features_cols = ["position", "area", "sex", "partner", "education"]
    cat_features_idx = [idx for idx, col in enumerate(train.columns) if
                        col in cat_features_cols]

    cat_boost_enc = ce.CatBoostEncoder(cols=cat_features_cols)

    train = data[data["is_test"] == 0]
    test = data[data["is_test"] == 1]
    cat_boost_enc.fit(train, target)
    train = cat_boost_enc.transform(train)
    test = cat_boost_enc.transform(test)

    train.drop(columns=["is_test"], inplace=True)
    test.drop(columns=["is_test"], inplace=True)

    predictions = lgb_cv(train, test, target,
                         model_params=model_params,
                         train_params=train_params,
                         cat_idx=cat_features_idx,
                         fold_schema=kf)


if __name__ == '__main__':
    main()
