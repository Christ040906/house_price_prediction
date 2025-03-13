import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.ensemble import HistGradientBoostingRegressor

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
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.array(X)  # 确保转换为numpy数组
        rooms_per_household = X[:, 3] / X[:, 4]
        population_per_household = X[:, 5] / X[:, 4]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 6] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]

# 主流程
def main():
    # 数据加载
    train_set, test_set = load_data()
    
    # 处理 median_income 异常值
    mean_income = train_set['median_income'].mean()
    std_income = train_set['median_income'].std()
    train_set = train_set[np.abs((train_set['median_income'] - mean_income) / std_income) < 4]
    
    # 训练数据处理
    X_train = train_set.drop("median_house_value", axis=1)
    y_train = train_set["median_house_value"].copy()
    
    # 选取数值特征并移除贡献低的变量
    num_attribs = ['longitude', 'latitude', 'median_income', 'total_rooms', 'households', 'population', 'total_bedrooms']
    cat_attribs = ["ocean_proximity"]
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
    ])
    
    # 数据预处理
    X_train_prepared = full_pipeline.fit_transform(X_train)
    
    # 训练 Bayesian Optimized 模型
    param_space = {
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'max_iter': Integer(100, 500),
        'max_depth': Integer(3, 10),
        'min_samples_leaf': Integer(1, 20)
    }
    
    opt = BayesSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_space,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    opt.fit(X_train_prepared, y_train)
    
    # 选择最佳模型
    best_model = opt.best_estimator_
    best_params = opt.best_params_
    best_rmse = np.sqrt(-opt.best_score_)
    
    print(f"最佳参数: {best_params}")
    print(f"最佳RMSE: {best_rmse:.2f}")
    
    # 预测测试数据
    X_test_prepared = full_pipeline.transform(test_set)
    final_predictions = best_model.predict(X_test_prepared)
    
    # 保存预测结果
    output = pd.DataFrame({
        "Id": test_set["Id"],
        "median_house_value": final_predictions
    })
    # output.to_csv("predictions.csv", index=False)
    # print("预测结果已保存至 predictions.csv")

if __name__ == "__main__":
    main()
