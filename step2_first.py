import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/raxhel/Downloads/games_details_new.csv'
games_details = pd.read_csv(file_path)

# 使用最新公式创建梦幻分数列
def calculate_fantasy_points(row):
    fantasy_points = (row['PTS'] * 1) + (row['REB'] * 1.2) + (row['AST'] * 1.5) + (row['STL'] * 3) + (row['BLK'] * 3) - (row['TO'] * 1) + (row['MIN_float'] * 0.1) + (row['FGM'] * 1) + (row['FGA'] * -0.5) + (row['FG_PCT'] * 2) + (row['FG3M'] * 1.5) + (row['FG3A'] * -0.5) + (row['FG3_PCT'] * 2) + (row['FTM'] * 1) + (row['FTA'] * -0.5) + (row['FT_PCT'] * 2) + (row['OREB'] * 1.5) + (row['DREB'] * 1.5) + (row['PF'] * -0.5) + (row['PLUS_MINUS'] * 0.2)
    return fantasy_points

# 将分钟转换为浮点值的函数
def convert_min_to_float(x):
    if isinstance(x, str):
        try:
            # 处理格式为 "xx.000000:yy" 的情况
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

# 移除包含NaN值的行
games_details = games_details.dropna(subset=['MIN_float', 'FGM_per_min', 'FGA_per_min', 'FTM_per_min', 'FTA_per_min', 'OREB_per_min', 'DREB_per_min'])

# 为模型选择特征
features = [
    'PTS', 'AST', 'REB', 'STL', 'BLK', 'TO',
    'FGM_per_min', 'FGA_per_min', 'FTM_per_min',
    'FTA_per_min', 'OREB_per_min', 'DREB_per_min'
]
X = games_details[features]
y = games_details['FantasyPoints']

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# 构建和训练线性回归模型
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# 使用交叉验证评估模型
linear_regression_validation = cross_validate(linear_regressor, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

# 对测试集进行预测
linear_regression_predictions = linear_regressor.predict(X_test)

# 计算均方误差(MSE)和R平方(R^2)值
linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
linear_regression_r2 = r2_score(y_test, linear_regression_predictions)

# 显示结果
print(f'均方误差 (MSE): {linear_regression_mse}')
print(f'R平方 (R^2): {linear_regression_r2}')

# 评估预测变量的重要性
importance = pd.DataFrame({
    'Feature': features,
    'Importance': linear_regressor.coef_
}).sort_values(by='Importance', ascending=False)
print(importance)

# 绘制预测值与实际值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, linear_regression_predictions, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Fantasy Points')
plt.ylabel('Predicted Fantasy Points')
plt.title('Actual vs. Predicted Fantasy Points')
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
plt.scatter(linear_regression_predictions, y_test - linear_regression_predictions, alpha=0.3, color='red')
plt.hlines(0, linear_regression_predictions.min(), linear_regression_predictions.max(), colors='black', linestyles='dashed')
plt.xlabel('Predicted Fantasy Points')
plt.ylabel('Residuals')
plt.title('Residuals of Linear Regression Predictions')
plt.show()

# 加载NBA薪水数据
nba_salaries_path =  '/Users/raxhel/Desktop/nba_salaries.csv'
nba_salaries = pd.read_csv(nba_salaries_path)
player_list = list(nba_salaries['Player Name'].unique())

# 整合玩家数据
col = pd.DataFrame(columns=['Player Name', 'PredictedFantasyPoints'])
for player in player_list:
    optimization_data_per_player = games_details.loc[games_details['PLAYER_NAME'] == player]
    if not optimization_data_per_player.empty:
        predicted_fantasy_points = optimization_data_per_player['FantasyPoints'].mean()
        new_row = pd.DataFrame({'Player Name': [player], 'PredictedFantasyPoints': [predicted_fantasy_points]})
        col = pd.concat([col, new_row], ignore_index=True)

# 合并预测的梦幻分数和薪水数据
nba_salaries = nba_salaries.merge(col, on='Player Name', how='left')
nba_salaries['Points/Salary Ratio'] = 1000 * nba_salaries['PredictedFantasyPoints'] / nba_salaries['Salary']

# 显示前5名球员的数据
print(nba_salaries.sort_values(by='PredictedFantasyPoints', ascending=False).head(5))

# 保存结果到CSV文件
nba_salaries.to_csv('/Users/raxhel/Downloads/nba_salaries_with_predictions.csv', index=False)
