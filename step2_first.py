import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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

# 構建和訓練線性回歸模型
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# 使用交叉驗證評估模型
linear_regression_validation = cross_validate(linear_regressor, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

# 對測試集進行預測
linear_regression_predictions = linear_regressor.predict(X_test)

# 計算均方誤差(MSE)和R平方(R^2)值
linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
linear_regression_r2 = r2_score(y_test, linear_regression_predictions)

# 顯示結果
print(f'均方誤差 (MSE): {linear_regression_mse}')
print(f'R平方 (R^2): {linear_regression_r2}')

# 評估預測變量的重要性
importance = pd.DataFrame({
    'Feature': features,
    'Importance': linear_regressor.coef_
}).sort_values(by='Importance', ascending=False)
print(importance)

# 繪製預測值與實際值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, linear_regression_predictions, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Fantasy Points')
plt.ylabel('Predicted Fantasy Points')
plt.title('Actual vs. Predicted Fantasy Points')
plt.show()

# 繪製殘差圖
plt.figure(figsize=(10, 6))
plt.scatter(linear_regression_predictions, y_test - linear_regression_predictions, alpha=0.3, color='red')
plt.hlines(0, linear_regression_predictions.min(), linear_regression_predictions.max(), colors='black', linestyles='dashed')
plt.xlabel('Predicted Fantasy Points')
plt.ylabel('Residuals')
plt.title('Residuals of Linear Regression Predictions')
plt.show()

# 繪製特徵重要性條形圖
plt.figure(figsize=(10, 6))
plt.barh(importance['Feature'], importance['Importance'], color='purple')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# 繪製學習曲線
train_sizes, train_scores, test_scores = learning_curve(linear_regressor, X, y, cv=5, scoring='neg_mean_squared_error')
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()

# 殘差的正態性檢查（Q-Q圖）
plt.figure(figsize=(10, 6))
sm.qqplot(y_test - linear_regression_predictions, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.show()

# 殘差的自相關檢查（ACF圖）
plt.figure(figsize=(10, 6))
sm.graphics.tsa.plot_acf(y_test - linear_regression_predictions, lags=40)
plt.title('ACF of Residuals')
plt.show()

# 加載NBA薪水數據
nba_salaries_path = '/Users/raxhel/Desktop/nba_salaries.csv'
nba_salaries = pd.read_csv(nba_salaries_path)
player_list = list(nba_salaries['Player Name'].unique())

# 整合玩家數據
col = pd.DataFrame(columns=['Player Name', 'PredictedFantasyPoints'])
for player in player_list:
    optimization_data_per_player = games_details.loc[games_details['PLAYER_NAME'] == player]
    if not optimization_data_per_player.empty:
        predicted_fantasy_points = optimization_data_per_player['FantasyPoints'].mean()
        new_row = pd.DataFrame({'Player Name': [player], 'PredictedFantasyPoints': [predicted_fantasy_points]})
        col = pd.concat([col, new_row], ignore_index=True)

# 合併預測的夢幻分數和薪水數據
nba_salaries = nba_salaries.merge(col, on='Player Name', how='left')
nba_salaries['Points/Salary Ratio'] = 1000 * nba_salaries['PredictedFantasyPoints'] / nba_salaries['Salary']

# 顯示前5名球員的數據
print(nba_salaries.sort_values(by='PredictedFantasyPoints', ascending=False).head(5))

# 保存結果到CSV文件
nba_salaries.to_csv('/Users/raxhel/Downloads/nba_salaries_with_predictions.csv', index=False)
