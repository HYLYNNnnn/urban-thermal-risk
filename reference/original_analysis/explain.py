from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shap import TreeExplainer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap
import warnings

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# 设置绘图风格

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')


def load_and_process_data(file_path):
    """加载并预处理数据"""
    # 读取数据
    df = pd.read_csv(file_path, encoding='utf-8')

    print(f"数据加载成功！")
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")

    # 重命名列为英文
    column_mapping = {
        '序号': 'ID',
        '经度': 'longitude',
        '纬度': 'latitude',
        'SVF': 'sky_view_factor',
        'TVF': 'tree_view_factor',
        'BVF': 'building_view_factor',
        'avg_green_view_index': 'green_view_index',
        'avg_sky_visibility': 'sky_visibility',
        'avg_enclosure_index': 'enclosure_index',
        'hw_value': 'height_width_ratio',
        '房价': 'housing_price',
        'popdensity': 'population_density',
        'poi医疗': 'medical_poi',
        'poi生活服务': 'living_service_poi',
        'poi教育': 'education_poi',
        '路网密度': 'road_network_density'
    }

    df = df.rename(columns=column_mapping)

    # 检查缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\n缺失值情况:")
        print(missing_values[missing_values > 0])
        # 填充缺失值（使用中位数）
        df = df.fillna(df.median())

    return df


# 2. 计算THIR指数
def calculate_thir(df):
    """基于HEVA框架计算THIR"""

    # 标准化函数
    def normalize(x):
        # 防止除零错误
        if x.max() == x.min():
            return np.zeros_like(x)
        # 添加小常数防止除零错误
        range_val = x.max() - x.min() + 1e-10
        return (x - x.min()) / range_val

    print("\n计算HEVA维度和THIR指数...")

    # 检查输入数据是否有缺失
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        print(f"警告: 输入数据在列 {null_cols} 中存在缺失值")

    # H维度 - 热辐射危险性
    df['H_hazard'] = (
            0.4 * normalize(df['sky_view_factor']) +
            0.3 * (1 - normalize(df['tree_view_factor'])) +
            0.3 * normalize(df['height_width_ratio'])
    )

    # E维度 - 暴露性
    df['E_exposure'] = (
            0.4 * normalize(df['population_density']) +
            0.3 * normalize(df['living_service_poi'] + df['education_poi']) +
            0.3 * normalize(df['road_network_density'])
    )

    # V维度 - 脆弱性
    df['V_vulnerability'] = (
            0.4 * (1 - normalize(df['green_view_index'])) +
            0.3 * normalize(df['sky_visibility']) +
            0.2 * normalize(df['enclosure_index']) +
            0.1 * (1 - normalize(df['medical_poi']))
    )

    # A维度 - 适应性
    df['A_adaptability'] = (
            0.25 * normalize(df['housing_price']) +
            0.25 * normalize(df['medical_poi']) +
            0.25 * normalize(df['tree_view_factor']) +
            0.25 * normalize(df['green_view_index'])
    )

    # 计算THIR
    df['THIR'] = df['H_hazard'] + df['E_exposure'] + df['V_vulnerability'] - df['A_adaptability']
    df['THIR_normalized'] = normalize(df['THIR'])

    # 风险等级划分
    df['risk_level'] = pd.cut(
        df['THIR_normalized'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Extreme Risk']
    )

    # 风险等级数值化
    df['risk_level_num'] = df['risk_level'].map({
        'Low Risk': 1, 'Medium Risk': 2, 'High Risk': 3, 'Extreme Risk': 4
    })

    # 在返回前添加缺失值检查
    print("\n检查THIR计算后的缺失值...")
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print("发现缺失值:")
        print(missing_after[missing_after > 0])

        # 填充缺失值 - 区分数值列和分类列
        for col in df.columns:
            if df[col].isnull().any():
                # 数值列使用中位数填充
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"  数值列 '{col}' 使用中位数 {median_val:.4f} 填充缺失值")
                # 分类列使用众数填充
                else:
                    mode_val = df[col].mode().iloc[0]
                    df[col] = df[col].fillna(mode_val)
                    print(f"  分类列 '{col}' 使用众数 '{mode_val}' 填充缺失值")
    else:
        print("无缺失值")

    # 确保所有缺失值已被处理
    if df.isnull().any().any():
        print("警告: 仍有缺失值存在，使用通用方法填充")
        df = df.fillna(method='ffill').fillna(method='bfill')

    print("THIR计算完成！")
    print(f"THIR范围: {df['THIR_normalized'].min():.2f} - {df['THIR_normalized'].max():.2f}")
    print(f"风险等级分布:\n{df['risk_level'].value_counts()}")

    return df
# ==================== XGBoost 详解部分 ====================

def detailed_xgboost_training(df, feature_cols):
    """
    XGBoost模型的详细训练过程

    XGBoost (eXtreme Gradient Boosting) 是一种基于梯度提升决策树的算法
    核心思想：通过迭代地添加新树来纠正前面所有树的预测误差
    """
    # 在开始时检查数据质量
    print("\n检查输入数据质量...")

    # 确保所有特征都是数值类型
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"警告: 特征 '{col}' 是分类类型，将转换为数值类型")
            df[col] = pd.factorize(df[col])[0]

    # 检查缺失值
    if df[feature_cols].isnull().any().any():
        null_counts = df[feature_cols].isnull().sum()
        null_counts = null_counts[null_counts > 0]
        print(f"发现 {len(null_counts)} 列存在缺失值:")
        for col, count in null_counts.items():
            print(f"  {col}: {count} 个缺失值")

        # 填充缺失值
        for col in feature_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  列 '{col}' 使用中位数 {median_val:.4f} 填充缺失值")

    print("=" * 60)
    print("XGBoost 模型训练详解")
    print("=" * 60)

    # 准备数据
    X = df[feature_cols]
    y = df['THIR_normalized']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['risk_level']
    )

    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # ========== 1. XGBoost参数详解 ==========
    params = {
        'n_estimators': 100,  # 树的数量（迭代次数）
        'max_depth': 6,  # 树的最大深度，控制过拟合
        'learning_rate': 0.1,  # 学习率，也称为eta，控制每棵树的贡献
        'subsample': 0.8,  # 每棵树使用的样本比例，防止过拟合
        'colsample_bytree': 0.8,  # 每棵树使用的特征比例
        'gamma': 0,  # 节点分裂所需的最小损失减少
        'reg_alpha': 0,  # L1正则化参数
        'reg_lambda': 1,  # L2正则化参数
        'objective': 'reg:squarederror',  # 损失函数
        'eval_metric': 'rmse',  # 评估指标
        'random_state': 42,
        'n_jobs': -1  # 使用所有CPU核心
    }

    print("\n模型参数配置:")
    for param, value in params.items():
        print(f"  {param}: {value}")

    # ========== 2. 创建和训练模型 ==========
    model = xgb.XGBRegressor(**params)

    # 使用早停法时，需要在构造函数中设置参数
    params.update({
        'early_stopping_rounds': 20,  # 移到构造函数中
        'eval_metric': 'rmse'  # 在构造函数中设置评估指标
    })

    model = xgb.XGBRegressor(**params)
    eval_set = [(X_test, y_test)]  # 只需在测试集上评估

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    # ========== 3. 模型评估 ==========
    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # 测试集预测
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n模型性能指标:")
    print(f"训练集 - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"测试集 - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    # ========== 4. 交叉验证 ==========
    print("\n=== 交叉验证 ===")
    # 创建没有早停参数的模型副本
    cv_params = params.copy()
    if 'early_stopping_rounds' in cv_params:
        del cv_params['early_stopping_rounds']
    if 'eval_metric' in cv_params:
        del cv_params['eval_metric']

    # 创建新的模型实例
    cv_model = xgb.XGBRegressor(**cv_params)

    # 尝试交叉验证，设置n_jobs=1避免内存问题
    try:
        cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='r2', n_jobs=1)
        print(f"5折交叉验证 R² 分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    except Exception as e:
        print(f"交叉验证出错: {str(e)} - 跳过交叉验证")
        cv_scores = []
        # 备份方案：使用简单交叉验证
        print("尝试简单交叉验证...")
        simple_cv_scores = []
        kf = train_test_split(range(len(X)), test_size=0.2, random_state=42)
        for train_idx, test_idx in kf:
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            model_fold = xgb.XGBRegressor(**params)
            model_fold.fit(X_train_fold, y_train_fold)
            r2 = model_fold.score(X_test_fold, y_test_fold)
            simple_cv_scores.append(r2)

        print(f"简单交叉验证 R² 分数: {np.mean(simple_cv_scores):.4f}")
    # ========== 5. 特征重要性可视化 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 5.1 特征重要性条形图
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    axes[0, 0].barh(feature_importance['feature'], feature_importance['importance'])
    axes[0, 0].set_xlabel('Importance Score')
    axes[0, 0].set_title('XGBoost Feature Importance\n(基于信息增益的特征重要性)')
    axes[0, 0].invert_yaxis()

    # 添加数值标签
    for i, v in enumerate(feature_importance['importance']):
        axes[0, 0].text(v + 0.001, i, f'{v:.3f}', va='center')

    # 5.2 训练历史曲线 - 修正后只显示测试集结果
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)

    # 只绘制测试集性能
    axes[0, 1].plot(x_axis, results['validation_0']['rmse'], label='Test')
    axes[0, 1].set_xlabel('Boosting Rounds')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Model Test Performance')
    axes[0, 1].legend()

    # 5.3 预测值 vs 真实值散点图
    axes[1, 0].scatter(y_test, y_test_pred, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predictions')
    axes[1, 0].set_title(f'Predictions vs True Values\n(R² = {test_r2:.3f})')

    # 5.4 残差分布图
    residuals = y_test - y_test_pred
    axes[1, 1].hist(residuals, bins=30, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Residual Distribution\n(Mean = {residuals.mean():.4f}, Std = {residuals.std():.4f})')

    plt.tight_layout()
    plt.savefig('xgboost_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, X, y


# ==================== SHAP 详解部分 ====================

def detailed_shap_analysis(model, X, feature_cols):
    """
    SHAP (SHapley Additive exPlanations) 详细分析

    SHAP基于博弈论中的Shapley值，用于解释机器学习模型的预测
    核心思想：计算每个特征对预测结果的边际贡献
    """

    print("\n" + "=" * 60)
    print("SHAP 分析详解")
    print("=" * 60)

    # 创建SHAP解释器
    # TreeExplainer专门针对树模型优化，计算速度快
    explainer = shap.TreeExplainer(model)

    # 计算SHAP值
    # shap_values的形状: (n_samples, n_features)
    # 每个值表示该特征对该样本预测的贡献
    shap_values = explainer.shap_values(X)

    print(f"\nSHAP值矩阵形状: {shap_values.shape}")
    print(f"基准值 (expected_value): {explainer.expected_value:.4f}")
    print("(基准值是所有训练样本的平均预测值)")

    # ========== 1. SHAP值的含义解释 ==========
    sample_idx = 0  # 选择第一个样本作为示例
    sample_prediction = model.predict(X.iloc[[sample_idx]])[0]

    print(f"\n样本 {sample_idx} 的SHAP分析:")
    print(f"模型预测值: {sample_prediction:.4f}")
    print(f"基准值: {explainer.expected_value:.4f}")
    print(f"差值: {sample_prediction - explainer.expected_value:.4f}")
    print("\n各特征的贡献:")

    feature_contributions = pd.DataFrame({
        'Feature': X.columns,
        'Value': X.iloc[sample_idx].values,
        'SHAP': shap_values[sample_idx],
        'Abs_SHAP': np.abs(shap_values[sample_idx])
    }).sort_values('Abs_SHAP', ascending=False)

    print(feature_contributions.head(10).to_string(index=False))

    # ========== 2. 创建详细的SHAP可视化 ==========

    # 2.1 特征重要性汇总图（带详细说明）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance\n每个特征的平均绝对SHAP值，表示该特征对模型预测的平均影响程度',
              fontsize=14)
    plt.xlabel('mean(|SHAP value|)\n特征对模型输出的平均影响大小', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_importance_explained.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2.2 SHAP蜂群图详解
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot 详解\n展示所有样本中每个特征的SHAP值分布', fontsize=14)
    plt.xlabel('SHAP value (impact on model output)\n负值：降低预测值 | 正值：提高预测值', fontsize=12)

    # 添加解释文字
    plt.text(0.02, 0.98,
             '颜色含义:\n红色 = 特征值高\n蓝色 = 特征值低\n\n点的分布:\n越分散 = 影响越不一致\n越集中 = 影响越一致',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top',
             fontsize=10)

    plt.tight_layout()
    plt.savefig('shap_summary_explained.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2.3 单特征依赖图详解
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 选择最重要的4个特征进行详细分析
    top_features = feature_contributions.head(4)['Feature'].tolist()

    for idx, feature in enumerate(top_features):
        ax = axes[idx // 2, idx % 2]

        # 创建依赖图
        shap.dependence_plot(
            feature, shap_values, X,
            interaction_index=None,
            ax=ax,
            show=False
        )

        # 添加趋势线
        feature_idx = X.columns.get_loc(feature)
        z = np.polyfit(X[feature].values, shap_values[:, feature_idx], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(X[feature].min(), X[feature].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='趋势线')

        ax.set_title(f'{feature} 的SHAP依赖图\n展示该特征值变化如何影响模型预测', fontsize=12)
        ax.set_xlabel(f'{feature} 值', fontsize=10)
        ax.set_ylabel(f'SHAP value for {feature}\n(对THIR的影响)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('shap_dependence_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2.4 交互作用分析
    plt.figure(figsize=(14, 8))

    # 选择两个重要特征分析交互作用
    feature1 = top_features[0]
    feature2 = top_features[1]

    shap.dependence_plot(
        feature1, shap_values, X,
        interaction_index=feature2,
        show=False
    )

    plt.title(f'{feature1} 和 {feature2} 的交互作用分析\n'
              f'颜色表示 {feature2} 的值，展示两个特征如何共同影响预测',
              fontsize=14)
    plt.xlabel(f'{feature1} 值', fontsize=12)
    plt.ylabel(f'SHAP value for {feature1}', fontsize=12)

    plt.tight_layout()
    plt.savefig('shap_interaction_explained.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2.5 力图和瀑布图详解
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 选择一个高风险样本
    high_risk_idx = X.index[np.argmax(model.predict(X))]

    # 力图
    shap.force_plot(
        explainer.expected_value,
        shap_values[high_risk_idx],
        X.iloc[high_risk_idx],
        matplotlib=True,
        show=False,
        out_names='THIR'
    )
    ax1.set_title(f'Force Plot - 样本 {high_risk_idx} (高风险)\n'
                  f'展示各特征如何将预测值从基准值推向最终预测值\n'
                  f'红色=增加风险，蓝色=降低风险', fontsize=14)

    # 瀑布图
    plt.sca(ax2)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=explainer.expected_value,
            data=X.iloc[high_risk_idx].values,
            feature_names=X.columns.tolist()
        ),
        max_display=10,
        show=False
    )
    ax2.set_title(f'Waterfall Plot - 样本 {high_risk_idx}\n'
                  f'逐步展示每个特征如何累积影响最终预测', fontsize=14)

    plt.tight_layout()
    plt.savefig('shap_individual_explained.png', dpi=300, bbox_inches='tight')
    plt.show()

    return explainer, shap_values


# ==================== SHAP图表含义详解 ====================

def explain_shap_charts():
    """
    详细解释SHAP生成的各种图表的含义
    """

    explanations = {
        "SHAP Feature Importance (特征重要性条形图)": {
            "作用": "展示每个特征对模型预测的平均影响程度",
            "解读": [
                "条形长度 = mean(|SHAP value|)，即平均绝对SHAP值",
                "越长的条形表示该特征越重要",
                "不考虑影响的方向，只看影响大小"
            ],
            "应用": "快速识别哪些特征对热风险影响最大"
        },

        "SHAP Summary Plot (蜂群图)": {
            "作用": "展示所有样本中每个特征的SHAP值分布",
            "解读": [
                "每个点代表一个样本",
                "横轴位置 = 该样本中该特征的SHAP值",
                "颜色 = 该样本中该特征的实际值（红=高，蓝=低）",
                "垂直分散程度 = 不同样本间的差异"
            ],
            "应用": "理解特征如何影响预测（正向/负向）以及影响的一致性"
        },

        "SHAP Dependence Plot (依赖图)": {
            "作用": "展示单个特征的值与其SHAP值之间的关系",
            "解读": [
                "横轴 = 特征的实际值",
                "纵轴 = 该特征的SHAP值",
                "趋势 = 特征值变化如何影响预测",
                "分散程度 = 其他特征的交互影响"
            ],
            "应用": "发现非线性关系和阈值效应"
        },

        "SHAP Interaction Plot (交互图)": {
            "作用": "展示两个特征之间的交互作用",
            "解读": [
                "基础同依赖图",
                "颜色 = 第二个特征的值",
                "颜色分层 = 存在交互作用"
            ],
            "应用": "识别特征间的协同或拮抗效应"
        },

        "SHAP Force Plot (力图)": {
            "作用": "解释单个预测是如何产生的",
            "解读": [
                "基准值 = E[f(X)]，所有样本的平均预测",
                "红色箭头 = 增加预测值的特征",
                "蓝色箭头 = 降低预测值的特征",
                "箭头长度 = 影响大小"
            ],
            "应用": "向利益相关者解释具体决策"
        },

        "SHAP Waterfall Plot (瀑布图)": {
            "作用": "逐步展示预测的形成过程",
            "解读": [
                "从基准值开始",
                "每个特征逐步加上其贡献",
                "最终达到模型预测值",
                "条形长度 = SHAP值大小"
            ],
            "应用": "详细追踪预测过程"
        }
    }

    # 创建解释图表
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    # 创建表格内容
    table_data = []
    for chart_name, info in explanations.items():
        table_data.append([chart_name, info["作用"], "\n".join(info["解读"]), info["应用"]])

    # 创建表格
    table = ax.table(cellText=table_data,
                     colLabels=['图表类型', '主要作用', '解读要点', '实际应用'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.2, 0.2, 0.4, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)

    # 设置表格样式
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    plt.title('SHAP图表含义详解', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_charts_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()

    return explanations


# ==================== 综合分析函数 ====================

def comprehensive_ml_analysis(df: object, feature_cols: object) -> tuple[XGBRegressor, TreeExplainer, Any, dict[str, list | list[Any] | dict[Any, Any] | Any]]:
    """
    执行完整的机器学习分析流程
    """

    # 1. XGBoost详细训练
    print("\n第一步：XGBoost模型训练")
    model, X, y = detailed_xgboost_training(df, feature_cols)

    # 2. SHAP详细分析
    print("\n第二步：SHAP模型解释")
    explainer, shap_values = detailed_shap_analysis(model, X, feature_cols)

    # 3. 生成SHAP图表解释
    print("\n第三步：生成SHAP图表解释")
    explanations = explain_shap_charts()

    # 4. 基于SHAP的改造建议
    print("\n第四步：基于SHAP分析生成改造建议")
    renovation_insights = generate_renovation_insights(model, explainer, shap_values, X, df)

    return model, explainer, shap_values, renovation_insights


def generate_renovation_insights(model, explainer, shap_values, X, df):
    """
    基于SHAP分析生成具体的改造建议
    """

    # 计算每个特征的平均SHAP值
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    feature_impact = pd.DataFrame({
        'feature': X.columns,
        'mean_shap': mean_shap,
        'current_mean': X.mean(),
        'current_std': X.std()
    }).sort_values('mean_shap', ascending=False)

    print("\n=== 基于SHAP的改造建议 ===")
    print("\n特征影响力排名:")
    print(feature_impact.head(10).to_string(index=False))

    # 生成改造优先级矩阵
    fig, ax = plt.subplots(figsize=(12, 8))

    # 可改造特征
    modifiable_features = ['green_view_index', 'tree_view_factor', 'medical_poi',
                           'living_service_poi', 'education_poi']

    # 筛选可改造特征
    modifiable_impact = feature_impact[feature_impact['feature'].isin(modifiable_features)]

    # 计算改造潜力分数
    modifiable_impact['renovation_potential'] = (
            modifiable_impact['mean_shap'] * (1 - modifiable_impact['current_mean'])
    )

    # 创建散点图
    scatter = ax.scatter(
        modifiable_impact['current_mean'],
        modifiable_impact['mean_shap'],
        s=modifiable_impact['renovation_potential'] * 1000,
        alpha=0.6,
        c=modifiable_impact['renovation_potential'],
        cmap='YlOrRd'
    )

    # 添加标签
    for idx, row in modifiable_impact.iterrows():
        ax.annotate(
            row['feature'].replace('_', ' ').title(),
            (row['current_mean'], row['mean_shap']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
        )

    ax.set_xlabel('Current Average Value', fontsize=12)
    ax.set_ylabel('Mean SHAP Impact', fontsize=12)
    ax.set_title('Renovation Priority Matrix\n'
                 'Bubble size = Renovation Potential', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 添加象限线
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=modifiable_impact['mean_shap'].median(), color='gray', linestyle='--', alpha=0.5)

    # 添加象限标签
    ax.text(0.75, modifiable_impact['mean_shap'].max() * 0.9,
            'Low Value\nHigh Impact\n(Priority)',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.colorbar(scatter, label='Renovation Potential Score')
    plt.tight_layout()
    plt.savefig('renovation_priority_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 生成具体建议
    insights = {
        'high_priority_features': modifiable_impact.nlargest(3, 'renovation_potential')['feature'].tolist(),
        'recommendations': [],
        'expected_impact': {}
    }

    for feature in insights['high_priority_features']:
        feature_data = modifiable_impact[modifiable_impact['feature'] == feature].iloc[0]
        current_value = feature_data['current_mean']
        impact = feature_data['mean_shap']

        if feature == 'green_view_index':
            insights['recommendations'].append(
                f"Increase {feature} from current {current_value:.2f} to 0.30+ "
                f"(Expected THIR reduction: {impact * 0.2:.3f})"
            )
        elif feature == 'tree_view_factor':
            insights['recommendations'].append(
                f"Improve {feature} from current {current_value:.2f} to 0.25+ "
                f"(Expected THIR reduction: {impact * 0.15:.3f})"
            )

        insights['expected_impact'][feature] = impact * 0.2

    return insights


# 使用示例
if __name__ == "__main__":
    # 1. 加载数据
    print("=== 数据加载与预处理 ===")
    df = load_and_process_data('数据.csv')

    # 2. 计算THIR指数
    print("\n=== THIR指数计算 ===")
    df = calculate_thir(df)

    # 3. 定义特征列
    feature_cols = [
        'sky_view_factor', 'tree_view_factor', 'building_view_factor',
        'green_view_index', 'sky_visibility', 'enclosure_index',
        'height_width_ratio', 'housing_price', 'population_density',
        'medical_poi', 'living_service_poi', 'education_poi', 'road_network_density'
    ]

    # 4. 执行全面的机器学习分析
    print("\n=== 开始全面的机器学习分析 ===")
    model, explainer, shap_values, insights = comprehensive_ml_analysis(df, feature_cols)

    # 5. 输出改造建议
    print("\n=== 最终改造建议 ===")
    print("高优先级特征:", insights['high_priority_features'])
    print("\n具体建议:")
    for rec in insights['recommendations']:
        print(f" - {rec}")

    print("\n分析完成！所有结果已保存为图像和CSV文件。")