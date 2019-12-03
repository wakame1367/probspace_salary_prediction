import numpy as np
from sklearn.preprocessing import LabelEncoder


def area_encode(_train, _test):
    area_encoder = LabelEncoder()
    area_encoder.fit(
        list(set(_train["area"].unique()) & set(_test["area"].unique())))
    _train["area"] = area_encoder.transform(_train["area"])
    _test["area"] = area_encoder.transform(_test["area"])
    return _train, _test


def age_clip(df):
    # replace (60 <= value) -> 60
    return np.clip(df['age'], 0, 60)


def get_area_code(df):
    area_code = 0
    area_cat = {1: ["北海道"],
                2: ["青森県", "岩手県", "秋田県", "宮城県", "山形県", "福島県"],
                3: ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"],
                4: ["山梨県", "長野県", "新潟県", "富山県", "石川県", "福井県", "静岡県", "愛知県",
                    "岐阜県"],
                5: ["三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"],
                6: ["鳥取県", "島根県", "岡山県", "広島県", "山口県"],
                7: ["香川県", "愛媛県", "徳島県", "高知県"],
                8: ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]}
    for k, v in area_cat.items():
        if df in v:
            area_code = k
    return area_code
