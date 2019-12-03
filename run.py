from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

dataset_path = Path("resources")
train_path = dataset_path / "train_data.csv"
test_path = dataset_path / "test_data.csv"


def submit(predictions):
    pd.DataFrame(
        {"id": range(len(predictions)), "y": np.expm1(predictions)}).to_csv(
        "submission.csv", index=False)


def lgb_cv(_train, _test, _target, lgb_params, cat_idx, fold_schema):
    oof = np.zeros(len(_train))
    predictions = np.zeros(len(_test))

    for i, (trn_idx, val_idx) in enumerate(fold_schema.split(_train)):
        print("fold: {}".format(i))
        trn_data = lgb.Dataset(_train.iloc[trn_idx],
                               label=_target.iloc[trn_idx])
        val_data = lgb.Dataset(_train.iloc[val_idx],
                               label=_target.iloc[val_idx])
        model = lgb.train(lgb_params,
                          trn_data,
                          valid_names=['train', 'valid'],
                          valid_sets=[trn_data, val_data],
                          categorical_feature=cat_idx,
                          verbose_eval=200)
        oof[val_idx] = model.predict(_train.iloc[val_idx],
                                     num_iteration=model.best_iteration)
        print(mean_absolute_error(np.expm1(_target.iloc[val_idx]),
                                  np.expm1(oof[val_idx])))
        predictions += model.predict(_test,
                                     num_iteration=model.best_iteration) / kf.n_splits
    print(mean_absolute_error(np.expm1(_target), np.expm1(oof)))

    return predictions


def main():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    #
    train['city'] = train['area'].apply(
        lambda x: 1 if '東京' in x or '大阪' in x else 0)

    #
    train[train['age'] >= 60]['age'] = 60

    n_splits = 4
    kf = KFold(n_splits=n_splits, random_state=71, shuffle=True)

    model_params = {
        'boosting_type': 'gbdt',
        'objective': 'fair',
        'metric': 'fair',
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
        'n_estimators': 20000,
        'verbose_eval': -1
    }

    # id is wasted col
    drop_cols = ["id", "service_length"]

    train.drop(columns=drop_cols, inplace=True)
    test.drop(columns=drop_cols, inplace=True)

    target_col = "salary"
    cat_features_cols = ["position", "area", "sex", "partner", "education"]

    target = train[target_col]
    train.drop(columns=[target_col], inplace=True)


if __name__ == '__main__':
    main()
