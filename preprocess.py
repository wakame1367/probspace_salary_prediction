from sklearn.preprocessing import LabelEncoder


def area_encode(_train, _test):
    area_encoder = LabelEncoder()
    area_encoder.fit(
        list(set(_train["area"].unique()) & set(_test["area"].unique())))
    _train["area"] = area_encoder.transform(_train["area"])
    _test["area"] = area_encoder.transform(_test["area"])
    return _train, _test
