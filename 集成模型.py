###################尚未调超参#######################
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

# 自定义特征转换器（保持与原始代码一致）
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix, self.bedrooms_ix, self.population_ix, self.household_ix = 3, 4, 5, 6
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.array(X)
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]

# 统一预处理流程
def data_preprocessing(train_set, test_set):
    # 处理median_income异常值（与原始代码一致）
    mean_income = train_set['median_income'].mean()
    std_income = train_set['median_income'].std()
    train_set = train_set[np.abs((train_set['median_income'] - mean_income) / std_income) < 4]

    # 定义特征列（保持与原始代码一致）
    num_attribs = ['longitude', 'latitude', 'median_income', 
                  'total_rooms', 'households', 'population', 'total_bedrooms']
    cat_attribs = ["ocean_proximity"]

    # 数值特征处理管道
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    # 完整预处理流程
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_attribs),
    ])

    # 准备训练数据
    X_train = train_set.drop("median_house_value", axis=1)
    y_train = train_set["median_house_value"].copy()
    
    # 预处理数据
    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(test_set)
    
    return X_train_prepared, y_train, X_test_prepared

# Stacking集成模型
def stacking_model():
    # 加载数据
    train = pd.read_csv("/home/majinrong/d2l-zh/housing-price-prediction-cupai-25/train.csv")
    test = pd.read_csv("/home/majinrong/d2l-zh/housing-price-prediction-cupai-25/test.csv")
    
    # 预处理数据
    X, y, X_test = data_preprocessing(train, test)
    
    # 初始化基模型（参数来自各模型的最佳参数）
    base_models = [
        ('rf', RandomForestRegressor(
            max_depth=5,
            n_estimators=200, 
            max_features=6, 
            random_state=42
        )),
        ('hgb', HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=300,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42
        )),
        ('xgb', XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=1.0,
            colsample_bytree=0.8,
            random_state=42
        ))
    ]
    
    # 初始化元模型
    meta_model = BayesianRidge()
    
    # 创建存储矩阵
    meta_train = np.zeros((X.shape[0], len(base_models)))
    meta_test = np.zeros((X_test.shape[0], len(base_models)))
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 第一层：基模型训练
    for model_idx, (name, model) in enumerate(base_models):
        print(f"\nTraining {name}...")
        fold_test_preds = []
        
        for train_idx, valid_idx in kf.split(X):
            # 划分训练集/验证集
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            # 克隆模型保证每次训练独立
            cloned_model = clone(model)
            cloned_model.fit(X_train, y_train)
            
            # 验证集预测
            valid_pred = cloned_model.predict(X_valid)
            meta_train[valid_idx, model_idx] = valid_pred
            
            # 测试集预测
            test_pred = cloned_model.predict(X_test)
            fold_test_preds.append(test_pred)
        
        # 平均各折的测试集预测
        meta_test[:, model_idx] = np.mean(fold_test_preds, axis=0)
        
        # 打印模型性能
        mse = mean_squared_error(y, meta_train[:, model_idx])
        print(f"{name} MSE: {mse:.2f} | RMSE: {np.sqrt(mse):.2f}")
    
    # 第二层：元模型训练
    print("\nTraining meta model...")
    meta_model.fit(meta_train, y)
    
    # 最终预测
    final_pred = meta_model.predict(meta_test)
    
    # 保存结果
    # output = pd.DataFrame({
    #     "Id": test["Id"],
    #     "median_house_value": final_pred
    # })
    # output.to_csv("stacking_predictions.csv", index=False)
    # print("\n预测结果已保存至 stacking_predictions.csv")

if __name__ == "__main__":
    stacking_model()
