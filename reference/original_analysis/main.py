import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 数据加载和预处理
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
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

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

    return df


# 3. 探索性数据分析
def explore_data_analysis(df):
    """数据探索和可视化"""

    # 基础统计
    print("\n=== Basic Statistics ===")
    print(f"Average THIR: {df['THIR_normalized'].mean():.3f}")
    print(f"High Risk Ratio: {(df['risk_level'].isin(['High Risk', 'Extreme Risk'])).mean() * 100:.1f}%")
    print("\nRisk Level Distribution:")
    print(df['risk_level'].value_counts())

    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. THIR分布
    axes[0, 0].hist(df['THIR_normalized'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('THIR Distribution')
    axes[0, 0].set_xlabel('THIR Normalized Value')
    axes[0, 0].set_ylabel('Frequency')

    # 2. 风险等级饼图
    risk_counts = df['risk_level'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, colors=colors, autopct='%1.1f%%')
    axes[0, 1].set_title('Risk Level Distribution')

    # 3. HEVA维度对比
    heva_means = df[['H_hazard', 'E_exposure', 'V_vulnerability', 'A_adaptability']].mean()
    axes[0, 2].bar(heva_means.index, heva_means.values,
                   color=['red', 'orange', 'yellow', 'green'])
    axes[0, 2].set_title('HEVA Dimensions Average Score')
    axes[0, 2].set_ylabel('Normalized Score')

    # 4. 关键特征箱线图
    key_features = ['sky_view_factor', 'tree_view_factor', 'green_view_index', 'population_density']
    df[key_features].boxplot(ax=axes[1, 0])
    axes[1, 0].set_title('Key Features Distribution')
    axes[1, 0].set_xticklabels(['SVF', 'TVF', 'GVI', 'Pop Density'], rotation=45)

    # 5. THIR与绿视率关系
    scatter = axes[1, 1].scatter(df['green_view_index'], df['THIR_normalized'],
                                 c=df['risk_level_num'], cmap='RdYlGn_r', alpha=0.6)
    axes[1, 1].set_xlabel('Green View Index')
    axes[1, 1].set_ylabel('THIR Value')
    axes[1, 1].set_title('THIR vs Green View Index')

    # 6. THIR与人口密度关系
    axes[1, 2].scatter(df['population_density'], df['THIR_normalized'],
                       c=df['risk_level_num'], cmap='RdYlGn_r', alpha=0.6)
    axes[1, 2].set_xlabel('Population Density')
    axes[1, 2].set_ylabel('THIR Value')
    axes[1, 2].set_title('THIR vs Population Density')

    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()


# 4. XGBoost模型训练
def train_model(df):
    """训练XGBoost模型"""

    # 准备特征
    feature_cols = [
        'sky_view_factor', 'tree_view_factor', 'building_view_factor',
        'green_view_index', 'sky_visibility', 'enclosure_index',
        'height_width_ratio', 'housing_price', 'population_density',
        'medical_poi', 'living_service_poi', 'education_poi', 'road_network_density'
    ]

    X = df[feature_cols]
    y = df['THIR_normalized']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 模型评估
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"\n=== Model Performance ===")
    print(f"Training R²: {train_score:.3f}")
    print(f"Testing R²: {test_score:.3f}")

    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Feature Importance Top 5 ===")
    print(feature_importance.head())

    return model, X, y, feature_cols


# 5. SHAP分析
def shap_analysis(model, X, feature_cols):
    """SHAP分析"""

    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 1. 特征重要性条形图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance', fontsize=16)
    plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. SHAP蜂群图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Feature Impact Distribution', fontsize=16)
    plt.xlabel('SHAP value (impact on model output)', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 依赖图 - 绿视率
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        'green_view_index',
        shap_values,
        X,
        interaction_index=None,
        show=False
    )
    plt.title('SHAP Dependence Plot - Green View Index', fontsize=16)
    plt.xlabel('Green View Index', fontsize=12)
    plt.ylabel('SHAP value for Green View Index', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_dependence_gvi.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 依赖图 - 树木覆盖率
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        'tree_view_factor',
        shap_values,
        X,
        interaction_index=None,
        show=False
    )
    plt.title('SHAP Dependence Plot - Tree View Factor', fontsize=16)
    plt.xlabel('Tree View Factor', fontsize=12)
    plt.ylabel('SHAP value for Tree View Factor', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_dependence_tvf.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. 交互作用图
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        'green_view_index',
        shap_values,
        X,
        interaction_index='tree_view_factor',
        show=False
    )
    plt.title('SHAP Interaction - Green View Index vs Tree View Factor', fontsize=16)
    plt.xlabel('Green View Index', fontsize=12)
    plt.ylabel('SHAP value for Green View Index', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_interaction_gvi_tvf.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. 单个样本分析
    # 选择一个高风险样本
    high_risk_idx = X.index[np.argmax(model.predict(X))]

    # 力图
    shap.force_plot(
        explainer.expected_value,
        shap_values[high_risk_idx],
        X.iloc[high_risk_idx],
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - Sample {high_risk_idx} (High Risk)', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_force_plot_example.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n=== SHAP Analysis Completed ===")
    print("Generated files:")
    print("1. shap_importance_bar.png")
    print("2. shap_summary_plot.png")
    print("3. shap_dependence_gvi.png")
    print("4. shap_dependence_tvf.png")
    print("5. shap_interaction_gvi_tvf.png")
    print("6. shap_force_plot_example.png")

    return explainer, shap_values


# 6. 数据质量检查
def check_data_quality(df, X):
    """检查数据质量"""
    print("\n=== Data Quality Check ===")
    print(f"Data shape: {X.shape}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    print(f"\nFeature ranges:")
    for col in X.columns:
        print(f"  {col}: [{X[col].min():.3f}, {X[col].max():.3f}]")


# 7. 改造需求评估
def assess_renovation_needs(df, model, explainer, feature_cols):
    """评估改造需求"""

    # 识别可改造特征
    modifiable_features = [
        'green_view_index', 'tree_view_factor', 'medical_poi', 'living_service_poi'
    ]

    # 计算每个点的改造潜力
    renovation_scores = []
    renovation_recommendations = []

    for idx, row in df.iterrows():
        X_single = row[feature_cols].values.reshape(1, -1)
        shap_single = explainer.shap_values(X_single)[0]

        # 计算改造潜力
        potential = 0
        recommendations = []

        # 绿视率改造潜力
        if row['green_view_index'] < 0.2:
            gvi_idx = feature_cols.index('green_view_index')
            if shap_single[gvi_idx] > 0:
                potential += abs(shap_single[gvi_idx]) * (0.4 - row['green_view_index'])
                recommendations.append('Increase vertical greening')

        # 树木覆盖改造潜力
        if row['tree_view_factor'] < 0.15:
            tvf_idx = feature_cols.index('tree_view_factor')
            if shap_single[tvf_idx] > 0:
                potential += abs(shap_single[tvf_idx]) * (0.3 - row['tree_view_factor'])
                recommendations.append('Plant more street trees')

        # 医疗设施改造潜力
        if row['medical_poi'] < 0.5:
            med_idx = feature_cols.index('medical_poi')
            potential += 0.05
            recommendations.append('Add medical facilities')

        # 判断是否需要改造
        needs_renovation = (
                row['THIR_normalized'] > 0.5 or
                potential > 0.1 or
                row['green_view_index'] < 0.15
        )

        # 确定改造优先级
        if row['THIR_normalized'] > 0.7 and potential > 0.15:
            priority = 'Urgent'
        elif row['THIR_normalized'] > 0.5 or potential > 0.1:
            priority = 'High'
        elif needs_renovation:
            priority = 'Medium'
        else:
            priority = 'Low'

        renovation_scores.append({
            'renovation_potential': potential,
            'needs_renovation': needs_renovation,
            'renovation_priority': priority,
            'recommendations': '; '.join(recommendations) if recommendations else 'Maintain current status'
        })

    # 添加到数据框
    renovation_df = pd.DataFrame(renovation_scores)
    df['renovation_potential'] = renovation_df['renovation_potential']
    df['needs_renovation'] = renovation_df['needs_renovation']
    df['renovation_priority'] = renovation_df['renovation_priority']
    df['renovation_recommendations'] = renovation_df['recommendations']

    return df


# 8. 输出结果
def export_results(df):
    """输出分析结果到CSV"""

    # 选择要输出的列
    output_columns = [
        'ID', 'longitude', 'latitude',
        'THIR', 'THIR_normalized', 'risk_level',
        'H_hazard', 'E_exposure', 'V_vulnerability', 'A_adaptability',
        'renovation_potential', 'needs_renovation', 'renovation_priority', 'renovation_recommendations',
        'sky_view_factor', 'tree_view_factor', 'green_view_index', 'population_density', 'housing_price'
    ]

    # 创建输出数据框
    output_df = df[output_columns].copy()

    # 保存到CSV
    output_file = 'thermal_risk_analysis_results.csv'
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n=== Results Saved ===")
    print(f"File name: {output_file}")
    print(f"Total records: {len(output_df)}")
    print(f"Points needing renovation: {output_df['needs_renovation'].sum()}")
    print(f"Urgent renovation points: {(output_df['renovation_priority'] == 'Urgent').sum()}")

    # 生成统计报告
    stats_report = f"""
    === Thermal Risk Analysis Report ===

    1. Overall Risk Status
    - Total points analyzed: {len(df)}
    - Average THIR: {df['THIR_normalized'].mean():.3f}
    - Standard deviation: {df['THIR_normalized'].std():.3f}

    2. Risk Level Distribution
    - Low Risk: {(df['risk_level'] == 'Low Risk').sum()} ({(df['risk_level'] == 'Low Risk').mean() * 100:.1f}%)
    - Medium Risk: {(df['risk_level'] == 'Medium Risk').sum()} ({(df['risk_level'] == 'Medium Risk').mean() * 100:.1f}%)
    - High Risk: {(df['risk_level'] == 'High Risk').sum()} ({(df['risk_level'] == 'High Risk').mean() * 100:.1f}%)
    - Extreme Risk: {(df['risk_level'] == 'Extreme Risk').sum()} ({(df['risk_level'] == 'Extreme Risk').mean() * 100:.1f}%)

    3. Renovation Analysis
    - Points needing renovation: {df['needs_renovation'].sum()} ({df['needs_renovation'].mean() * 100:.1f}%)
    - Average renovation potential: {df['renovation_potential'].mean():.3f}
    - Priority distribution:
      * Urgent: {(df['renovation_priority'] == 'Urgent').sum()}
      * High: {(df['renovation_priority'] == 'High').sum()}
      * Medium: {(df['renovation_priority'] == 'Medium').sum()}
      * Low: {(df['renovation_priority'] == 'Low').sum()}

    4. Key Issues
    - Low green view index (<0.2): {(df['green_view_index'] < 0.2).sum()} points
    - Low tree coverage (<0.15): {(df['tree_view_factor'] < 0.15).sum()} points
    - High population density: {(df['population_density'] > df['population_density'].mean() + df['population_density'].std()).sum()} points
    """

    # 保存统计报告
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(stats_report)

    print(stats_report)

    return output_df


# 9. 创建空间分布图
def create_spatial_visualization(df):
    """创建空间分布可视化"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. THIR空间分布
    scatter1 = axes[0, 0].scatter(
        df['longitude'], df['latitude'],
        c=df['THIR_normalized'],
        cmap='RdYlGn_r',
        s=100, alpha=0.8,
        edgecolors='black', linewidth=0.5
    )
    axes[0, 0].set_title('THIR Spatial Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(scatter1, ax=axes[0, 0], label='THIR Value')

    # 2. 风险等级分布
    risk_colors = {'Low Risk': 'green', 'Medium Risk': 'yellow', 'High Risk': 'orange', 'Extreme Risk': 'red'}
    for risk, color in risk_colors.items():
        mask = df['risk_level'] == risk
        axes[0, 1].scatter(
            df.loc[mask, 'longitude'],
            df.loc[mask, 'latitude'],
            c=color, label=risk, s=80, alpha=0.7,
            edgecolors='black', linewidth=0.5
        )
    axes[0, 1].set_title('Risk Level Spatial Distribution', fontsize=14)
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].legend()

    # 3. 改造需求分布
    need_renovation = df['needs_renovation']
    axes[1, 0].scatter(
        df.loc[need_renovation, 'longitude'],
        df.loc[need_renovation, 'latitude'],
        c='red', label='Needs Renovation', s=100, alpha=0.7, marker='^'
    )
    axes[1, 0].scatter(
        df.loc[~need_renovation, 'longitude'],
        df.loc[~need_renovation, 'latitude'],
        c='green', label='No Renovation Needed', s=50, alpha=0.5
    )
    axes[1, 0].set_title('Renovation Needs Distribution', fontsize=14)
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].legend()

    # 4. 改造潜力分布
    scatter4 = axes[1, 1].scatter(
        df['longitude'], df['latitude'],
        c=df['renovation_potential'],
        cmap='YlOrRd',
        s=100, alpha=0.8,
        edgecolors='black', linewidth=0.5
    )
    axes[1, 1].set_title('Renovation Potential Spatial Distribution', fontsize=14)
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    plt.colorbar(scatter4, ax=axes[1, 1], label='Renovation Potential')

    plt.tight_layout()
    plt.savefig('spatial_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主程序
def main():
    """主程序入口"""

    print("=== Thermal Risk Analysis Started ===\n")

    # 1. 加载数据
    df = load_and_process_data('数据.csv')

    # 2. 计算THIR
    df = calculate_thir(df)

    # 3. 数据探索
    explore_data_analysis(df)

    # 4. 训练模型
    model, X, y, feature_cols = train_model(df)

    # 5. 检查数据质量
    check_data_quality(df, X)

    # 6. SHAP分析
    explainer, shap_values = shap_analysis(model, X, feature_cols)

    # 7. 评估改造需求
    df = assess_renovation_needs(df, model, explainer, feature_cols)

    # 8. 创建空间可视化
    create_spatial_visualization(df)

    # 9. 输出结果
    output_df = export_results(df)

    print("\n=== Analysis Completed! ===")
    print("Output files:")
    print("1. thermal_risk_analysis_results.csv - Complete analysis results")
    print("2. analysis_report.txt - Statistical report")
    print("3. data_exploration.png - Data exploration charts")
    print("4. shap_*.png - SHAP analysis charts")
    print("5. spatial_distribution.png - Spatial distribution maps")

    return df, model, output_df


# 运行主程序
if __name__ == "__main__":
    df, model, output_df = main()