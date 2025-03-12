import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 数据加载与特征工程
def load_data():
    train_path = "/home/majinrong/d2l-zh/housing-price-prediction-cupai-25/train.csv"
    test_path = "/home/majinrong/d2l-zh/housing-price-prediction-cupai-25/test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# 自定义特征转换器
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix, self.bedrooms_ix, self.population_ix, self.household_ix = 3, 4, 5, 6
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.array(X)  # 确保转换为numpy数组
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]

# 主流程
def main():
    # 数据加载
    train_set, test_set = load_data()
    print(test_set.head())
    # 训练数据处理
    X_train = train_set.drop("median_house_value", axis=1)
    y_train = train_set["median_house_value"].copy()

    # 构建预处理管道
    num_attribs = list(X_train.select_dtypes(include=[np.number]).columns)
    num_attribs.remove("ocean_proximity") if "ocean_proximity" in num_attribs else None
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    # 模型训练
    X_train_prepared = full_pipeline.fit_transform(X_train)
    
    param_grid = [
        {'n_estimators': [30, 50], 'max_features': [6, 8]},
        {'bootstrap': [False], 'n_estimators': [15, 30], 'max_features': [4, 6]}
    ]
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train_prepared, y_train)

    # 测试数据预测
    X_test_prepared = full_pipeline.transform(test_set)
    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(X_test_prepared)
    
    # 保存预测结果
    output = pd.DataFrame({
        "Id": test_set["Id"],
        "median_house_value": final_predictions
    })
    output.to_csv("predictions.csv", index=False)
    print("预测结果已保存至 predictions.csv")

if __name__ == "__main__":
    main()
