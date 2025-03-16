import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
import logging
from time import time
import lightgbm as lgbm

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

    # 定义各基模型的参数网格
    xgb_params = {
        'n_estimators': [900],
        'learning_rate': [0.07],
        'max_depth': [5]
    }
    cat_params = {
        'iterations': [1000],
        'learning_rate': [0.1],
        'depth': [8]
    }
    hgb_params = {
        'max_iter': [500],
        'learning_rate': [0.09],
        'max_depth': [10]
    }
    lgbm_params = {
        'colsample_bytree': [0.55],
        'learning_rate': [0.3],
        'max_depth': [13],
        'n_estimators': [1000]

    }
    
    # 调优各基模型
    best_xgb = tune_model(X_train_prepared, y_train, XGBRegressor(random_state=42), xgb_params, 'XGBoost')
    best_cat = tune_model(X_train_prepared, y_train, CatBoostRegressor(random_state=42, verbose=0), cat_params, 'CatBoost')
    best_hgb = tune_model(X_train_prepared, y_train, HistGradientBoostingRegressor(random_state=42), hgb_params, 'HGBoost')
    best_lgbm = tune_model(X_train_prepared, y_train, lgbm.LGBMRegressor(random_state=42), lgbm_params, 'LightGBM')
    
    # 构建Stacking集成模型
    estimators = [
        ('xgb', best_xgb),
        ('cat', best_cat),
        ('hgb', best_hgb),
        ('lgbm', best_lgbm)
    ]
    meta_model = BayesianRidge()

    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    logging.info("\n=== 开始训练 Stacking 模型 ===")
    start = time()
    stack_reg.fit(X_train_prepared, y_train)
    logging.info(f"Stacking 模型训练完毕, 耗时: {time()-start:.1f}s")
    
    # 对测试数据进行预测
    final_predictions = stack_reg.predict(X_test_prepared)
    
    # 保存预测结果
    output = pd.DataFrame({
        "Id": test_set["Id"],
        "median_house_value": final_predictions
    })
    output.to_csv("predictions.csv", index=False)
    logging.info("预测结果已保存至 predictions.csv")

if __name__ == "__main__":
    main()
