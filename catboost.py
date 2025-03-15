import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
import logging
from time import time

# 配置日志
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("hparam_tuning.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("初始化完成，开始执行流程")

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

# 高效超参数调优函数，使用GridSearchCV
def tune_model(X, y, model, params, model_name):
    start = time()
    logging.info(f"\n=== 开始调试 {model_name} ===")
    
    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    best_rmse = np.sqrt(-grid.best_score_)
    logging.info(f"{model_name} 最佳参数: {grid.best_params_}")
    logging.info(f"{model_name} 最佳RMSE: {best_rmse:.2f}")
    logging.info(f"{model_name} 调试耗时: {time()-start:.1f}s")
    
    return grid.best_estimator_

def main():
    setup_logger()
    
    # 数据加载
    train_set, test_set = load_data()
    logging.info("测试集前五行：")
    logging.info(test_set.head())
    
    # 分离特征和标签
    X_train = train_set.drop("median_house_value", axis=1)
    y_train = train_set["median_house_value"].copy()

    # 构建预处理管道
    num_attribs = list(X_train.select_dtypes(include=[np.number]).columns)
    if "ocean_proximity" in num_attribs:
        num_attribs.remove("ocean_proximity")
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

    # 对训练数据和测试数据进行预处理
    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(test_set)

    # 定义 CatBoostRegressor 的参数网格
    cat_params = {
        'iterations': [400,500,600,700,800,900],
        'learning_rate': [0.15,0.2,0.25,0.3,0.35,0.4],
        'depth': [3, 5, 7,9,11,13,15,17]
    }
    
    # 调优 CatBoostRegressor 模型
    best_cat = tune_model(X_train_prepared, y_train, CatBoostRegressor(random_state=42, verbose=0), cat_params, 'CatBoost')
    
    # 对测试数据进行预测
    final_predictions = best_cat.predict(X_test_prepared)
    
    # 保存预测结果
    output = pd.DataFrame({
        "Id": test_set["Id"],
        "median_house_value": final_predictions
    })
    output.to_csv("predictions-cat.csv", index=False)
    logging.info("预测结果已保存至 predictions.csv")

if __name__ == "__main__":
    main()
