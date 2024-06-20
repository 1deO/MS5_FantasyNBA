import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/raxhel/Desktop/games_details_new.csv'
games_details = pd.read_csv(file_path)

# 使用最新公式创建梦幻分数列
def calculate_fantasy_points(row):
    fantasy_points = (row['PTS'] * 1) + (row['REB'] * 1.2) + (row['AST'] * 1.5) + (row['STL'] * 3) + (row['BLK'] * 3) - (row['TO'] * 1) + (row['MIN_float'] * 0.1) + (row['FGM'] * 1) + (row['FGA'] * -0.5) + (row['FG_PCT'] * 2) + (row['FG3M'] * 1.5) + (row['FG3A'] * -0.5) + (row['FG3_PCT'] * 2) + (row['FTM'] * 1) + (row['FTA'] * -0.5) + (row['FT_PCT'] * 2) + (row['OREB'] * 1.5) + (row['DREB'] * 1.5) + (row['PF'] * -0.5) + (row['PLUS_MINUS'] * 0.2)
    return fantasy_points

# 将分钟转换为浮点值的函数
def convert_min_to_float(x):
    if isinstance(x, str):
        try:
            parts = x.split(':')
            if len(parts) == 2:
                mins = float(parts[0])
                secs = int(parts[1])
                return mins + secs / 60
            else:
                return float(parts[0])
        except Exception as e:
            print(f"Error converting {x}: {e}")
            return np.nan
    elif isinstance(x, float):
        return x
    else:
        return np.nan

# 应用转换函数到MIN列
games_details['MIN_float'] = games_details['MIN'].apply(convert_min_to_float)

# 计算梦幻分数
games_details['FantasyPoints'] = games_details.apply(calculate_fantasy_points, axis=1)

# 基于现有数据创建附加特征
games_details['FGM_per_min'] = games_details['FGM'] / games_details['MIN_float']
games_details['FGA_per_min'] = games_details['FGA'] / games_details['MIN_float']
games_details['FTM_per_min'] = games_details['FTM'] / games_details['MIN_float']
games_details['FTA_per_min'] = games_details['FTA'] / games_details['MIN_float']
games_details['OREB_per_min'] = games_details['OREB'] / games_details['MIN_float']
games_details['DREB_per_min'] = games_details['DREB'] / games_details['MIN_float']

# 移動平均計算
horizon = 3
for column_name in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TO', 'FGM_per_min', 'FGA_per_min', 'FTM_per_min', 'FTA_per_min', 'OREB_per_min', 'DREB_per_min', 'FantasyPoints']:
    games_details['moving_' + column_name] = games_details.groupby(['PLAYER_NAME'])[column_name].transform(lambda x: x.rolling(horizon, 1).mean().shift(1))

# 移除包含NaN值的行
games_details = games_details.dropna(subset=['MIN_float', 'FGM_per_min', 'FGA_per_min', 'FTM_per_min', 'FTA_per_min', 'OREB_per_min', 'DREB_per_min', 'moving_PTS'])

# 繪製相關矩陣的熱力圖
plt.figure(figsize=(15, 8))
correlation_matrix = games_details[['moving_PTS', 'moving_AST', 'moving_REB', 'moving_STL', 'moving_BLK', 'moving_TO', 'moving_FGM_per_min', 'moving_FGA_per_min', 'moving_FTM_per_min', 'moving_FTA_per_min', 'moving_OREB_per_min', 'moving_DREB_per_min', 'moving_FantasyPoints']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# 繪製直方圖
plt.figure(figsize=(14, 5))
sns.histplot(games_details['FantasyPoints'], color="blue", label="Fantasy Points", kde=True, stat="density", linewidth=0, bins=20)
plt.xlabel("Fantasy Points", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title('Histogram of Fantasy Points')
plt.legend()
plt.show()

# 為模型選擇特徵
features = [
    'moving_PTS', 'moving_AST', 'moving_REB', 'moving_STL', 'moving_BLK', 'moving_TO',
    'moving_FGM_per_min', 'moving_FGA_per_min', 'moving_FTM_per_min',
    'moving_FTA_per_min', 'moving_OREB_per_min', 'moving_DREB_per_min'
]
X = games_details[features]
y = games_details['moving_FantasyPoints']

# 將數據集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# 標準化數據
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso回歸
lasso_regressor = Lasso(alpha=0.01)
lasso_regressor.fit(X_train_scaled, y_train)

# Lasso預測
lasso_predictions = lasso_regressor.predict(X_test_scaled)

# 計算均方誤差(MSE)和R平方(R^2)值
lasso_mse = mean_squared_error(y_test, lasso_predictions)
lasso_r2 = r2_score(y_test, lasso_predictions)

# 顯示結果
print(f'Lasso均方誤差 (MSE): {lasso_mse}')
print(f'Lasso R平方 (R^2): {lasso_r2}')

# 評估Lasso選擇的變量的重要性
lasso_importance = pd.DataFrame({
    'Feature': features,
    'Importance': lasso_regressor.coef_
}).sort_values(by='Importance', ascending=False)
print(lasso_importance)

# 繪製Lasso特徵重要性條形圖
plt.figure(figsize=(10, 6))
plt.barh(lasso_importance['Feature'], lasso_importance['Importance'], color='purple')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Lasso Feature Importance')
plt.show()
