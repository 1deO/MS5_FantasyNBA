import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import gurobipy as gp

# Load the dataset
file_path = 'games_details.csv'
games_details = pd.read_csv(file_path)

# 使用最新公式创建梦幻分数列
def calculate_fantasy_points(row):
    fantasy_points = (row['PTS'] * 1) + (row['REB'] * 1) + (row['AST'] * 1.2) + (row['STL'] * 1) + (row['BLK'] * 1.5) - (row['TO'] * 1) + (row['FG3M'] * 0.5) + (row['FGM'] * 0.3) - (row['PF'] * 3)
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

print(games_details.head())

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
print(f"Total number of games_details: {len(games_details)}")

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

forecasting_data = games_details[games_details['GAME_DATE_EST'] != '2022-12-22']
print(forecasting_data.shape)

# 為模型選擇特徵
features = [
    'moving_PTS', 'moving_AST', 'moving_REB', 'moving_STL', 'moving_BLK', 'moving_TO',
    'moving_FGM_per_min', 'moving_FGA_per_min', 'moving_FTM_per_min',
    'moving_FTA_per_min', 'moving_OREB_per_min', 'moving_DREB_per_min'
]
X = forecasting_data[features]
y = forecasting_data['FantasyPoints']
print(f"Total number of games_details[features]: {len(forecasting_data[features])}")
print(f"Total number of games_details['FantasyPoints']: {len(forecasting_data['FantasyPoints'])}")
print(y)

# 將數據集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# 構建和訓練線性回歸模型
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# 使用交叉驗證評估模型
linear_regression_validation = cross_validate(linear_regressor, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)
linear_regression_validation['test_score']

# 對測試集進行預測
linear_regression_predictions = linear_regressor.predict(X_test)
print(f"Predictions: {linear_regression_predictions[:5]}")

print(f"Total number of X_test: {len(X_test)}")
print(f"Total number of predictions: {len(linear_regression_predictions)}")

# 檢查模型性能
linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
linear_regression_r2 = r2_score(y_test, linear_regression_predictions)
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

LR_final = linear_regressor
LR_final.fit(X, y)

optimization_dataset = games_details
optimization_dataset['PredictedFantasyPoints'] = LR_final.predict(games_details[features])
print(optimization_dataset[['PLAYER_NAME', 'PredictedFantasyPoints']].head())

# 加載NBA薪水數據
nba_salaries_path = 'nba_salaries.csv'
nba_salaries = pd.read_csv(nba_salaries_path)

# 加載NBA薪水數據
nba_salaries_path = 'nba_salaries.csv'
nba_salaries = pd.read_csv(nba_salaries_path)
player_list = list(nba_salaries['Player Name'].unique())
print(player_list)
print(len(player_list))
col = pd.DataFrame()
print(col)

# 加载比赛详情数据
games_details_path = 'games_details.csv'
games_details = pd.read_csv(games_details_path)

# 确保比赛日期列是datetime格式
games_details['GAME_DATE_EST'] = pd.to_datetime(games_details['GAME_DATE_EST'])

# 筛选出在2022-12-20比赛中的球员
filtered_games_details = games_details[games_details['GAME_DATE_EST'] == '2022-12-22']

# 获取在该日期打球的球员名称
players_in_game = filtered_games_details['PLAYER_NAME'].unique()

# 创建一个DataFrame来存储结果
players_df = pd.DataFrame(players_in_game, columns=['Player Name'])

# 合并两个数据集以获取球员薪水
players_salaries = pd.merge(players_df, nba_salaries, left_on='Player Name', right_on='Player Name', how='left')

# 删除Salary为空值的行
players_salaries = players_salaries.dropna(subset=['Salary'])

# 打印结果
print(players_salaries[['Player Name', 'Salary']])

# 合併預測的夢幻分數和薪水數據，加入 Position 欄位
nba_salaries = nba_salaries.merge(
    optimization_dataset[['PLAYER_NAME', 'PredictedFantasyPoints']],
    left_on='Player Name',
    right_on='PLAYER_NAME',
    how='left'
)

# 將每個球員的夢幻分數整合成平均值，並加入 Position 欄位
nba_salaries_grouped = nba_salaries.groupby(['Player Name', 'Position']).agg({
    'PredictedFantasyPoints': 'mean',
    'Salary': 'first'  # 假設每個球員的薪水是一致的
}).reset_index()

nba_salaries_grouped['Points/Salary Ratio'] = 1000 * nba_salaries_grouped['PredictedFantasyPoints'] / nba_salaries_grouped['Salary']

# 顯示前5名球員的數據
print(nba_salaries_grouped.sort_values(by='PredictedFantasyPoints', ascending=False).head(5))

# 保存結果到CSV文件
output_file_path = 'nba_salaries_with_predictions.csv'
nba_salaries_grouped.to_csv(output_file_path, index=False)

# 移除包含空值的行並保存修改
nba_salaries_grouped.dropna(subset=['Player Name', 'PredictedFantasyPoints', 'Salary'], inplace=True)

# 印出清理後的 DataFrame
print(nba_salaries_grouped[['Player Name', 'Position', 'PredictedFantasyPoints', 'Salary']])

# 優化部分
indices = nba_salaries_grouped['Player Name']
points = dict(zip(indices, nba_salaries_grouped['PredictedFantasyPoints']))
salaries = dict(zip(indices, nba_salaries_grouped['Salary']))

S = 25000000  # 预算限制
m = gp.Model()

y = m.addVars(nba_salaries_grouped['Player Name'], vtype=gp.GRB.BINARY, name="y")

# 設定目標函數：最大化梦幻分数
m.setObjective(gp.quicksum(points[i] * y[i] for i in indices), gp.GRB.MAXIMIZE)

# 增加预算约束
m.addConstr(gp.quicksum(salaries[i] * y[i] for i in indices) <= S, name="salary")

# 增加每个位置的约束，确保每个位置至少选一名球员，并确保从四个位置中选三个位置
positions = ['PG', 'SG', 'PF', 'C']
position_vars = {pos: m.addVar(vtype=gp.GRB.BINARY, name=f"position_{pos}") for pos in positions}

# 确保每个位置至少有一名球员
for pos in positions:
    m.addConstr(gp.quicksum(y[i] for i in indices if nba_salaries_grouped.loc[nba_salaries_grouped['Player Name'] == i, 'Position'].values[0] == pos) >= position_vars[pos], name=f"min_{pos}")

# 确保从四个位置中选三个位置
m.addConstr(gp.quicksum(position_vars[pos] for pos in positions) == 3, name="select_3_positions")

# 确保每个位置最多有一个球员
for pos in positions:
    m.addConstr(gp.quicksum(y[i] for i in indices if nba_salaries_grouped.loc[nba_salaries_grouped['Player Name'] == i, 'Position'].values[0] == pos) <= 1, name=f"max_{pos}")

# 确保至少选择两名SF球员
m.addConstr(gp.quicksum(y[i] for i in indices if nba_salaries_grouped.loc[nba_salaries_grouped['Player Name'] == i, 'Position'].values[0] == 'SF') >= 2, name='SF')

# 增加球员数量限制
m.addConstr(gp.quicksum(y[i] for i in indices) <= 5, name="max_players")

# 确保所选的球员都在打印结果的名单中
m.addConstr(gp.quicksum(y[i] for i in indices) == gp.quicksum(y[i] for i in players_salaries['Player Name']), name="selected_players")

m.addConstr(gp.quicksum(points[i] * y[i] for i in indices)>=50)
# 优化模型
m.optimize()

# 检查模型状态并输出结果
if m.status == gp.GRB.OPTIMAL:
    selected_players = [v.varName for v in m.getVars() if v.x > 0 and v.varName.startswith('y')]
    print(f"Selected Players: {selected_players}")

    selected_player_names = [player.split('[')[-1][:-1] for player in selected_players]  # 获取球员名字

    # 打印被选中的球员并获取预测分数和实际分数
    predicted_scores = []
    actual_scores = []
    for player_name in selected_player_names:
        player_data_grouped = nba_salaries_grouped[nba_salaries_grouped['Player Name'] == player_name]
        predicted_score = player_data_grouped['PredictedFantasyPoints'].values[0]
        predicted_scores.append(predicted_score)

        player_data = nba_salaries[nba_salaries['Player Name'] == player_name]
        actual_score = games_details[(games_details['PLAYER_NAME'] == player_name) & (games_details['GAME_DATE_EST'] == '2022-12-22')]['PTS'].values[0]
        actual_scores.append(actual_score)

        print(f"Player: {player_name}, Position: {player_data_grouped['Position'].values[0]}, PredictedFantasyPoints: {predicted_score}, Salary: {player_data_grouped['Salary'].values[0]}")

    # 绘制条形图
    x = np.arange(len(selected_player_names))  # 球员数量
    width = 0.35  # 条形图的宽度

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, predicted_scores, width, label='Predicted Scores')
    bars2 = ax.bar(x + width/2, actual_scores, width, label='Actual Scores')

    # 添加标签和标题
    ax.set_xlabel('Player Names')
    ax.set_ylabel('Scores')
    ax.set_title('Predicted vs Actual Scores for Selected Players')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_player_names)
    ax.legend()

    # 在每个条形图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()
else:
    print("The model is infeasible; no optimal solution found.")


import matplotlib.pyplot as plt

# 保存结果数据框
#output_file_path = '/mnt/data/2023_player_scores_with_difference.csv'
#nba_scores &#8203;:citation[oaicite:0]{index=0}&#8203;

#Start
# 计算残差和 LSig
data = pd.DataFrame()
data['residuals']  = optimization_dataset['PTS'] - optimization_dataset['PredictedFantasyPoints']
print("Residuals:")
print(data['residuals'].head(10))

# Calculate LSig = ln(squared residuals)
data['LSig'] = np.log(data['residuals'] ** 2)
print("LSig Residuals:")
print(data['LSig'].head(10))

# 使用LSig进行线性回归并预测
linear_regressor_lsig = LinearRegression()
linear_regressor_lsig.fit(X, data['LSig'])
predicted_lsig = linear_regressor_lsig.predict(X)
data['Predicted_LSig'] = predicted_lsig
print("Regression done LSig Residuals:")
print(data['Predicted_LSig'].head(10))

# 模拟分数误差
S = 1000  # 模拟次数
simulated_scores = []

std_dev = np.sqrt(np.exp(data['Predicted_LSig']))

for i in range(S):
    simulated_error = np.random.normal(
        loc=0,
        scale=std_dev,
        size=len(data)
    )
    simulated_score = optimization_dataset['PredictedFantasyPoints'] + simulated_error
    simulated_scores.append(simulated_score)

simulated_scores = np.array(simulated_scores).T  # 转置以匹配原始数据的形状
print(optimization_dataset['PredictedFantasyPoints'][:10])
print(simulated_scores[:10])

# 计算每个球员的平均模拟分数
average_simulated_scores = np.mean(simulated_scores, axis=1)

# 将平均模拟分数添加到原始数据集中
optimization_dataset['AverageSimulatedScore'] = average_simulated_scores

# 根据球员姓名计算每个球员的平均分数
player_avg_scores = optimization_dataset.groupby('PLAYER_NAME')['AverageSimulatedScore'].mean().reset_index()

# 将有平均分数的资料加入到nba_salaries中，确保包含球员姓名、薪水、预测的平均分数、位置
nba_salaries = nba_salaries.merge(
    player_avg_scores,
    left_on='Player Name',
    right_on='PLAYER_NAME',
    how='left'
).dropna(subset=['AverageSimulatedScore'])

# 删除重复的球员记录
nba_salaries = nba_salaries.drop_duplicates(subset=['Player Name'])

# 确保保留需要的字段
nba_salaries = nba_salaries[['Player Name', 'Salary', 'AverageSimulatedScore', 'Position']]
print("nba_salary:")
print(nba_salaries.head())
# 过滤比赛日期为2022-12-22的球员
games_details['GAME_DATE_EST'] = pd.to_datetime(games_details['GAME_DATE_EST'])
filtered_games_details = games_details[games_details['GAME_DATE_EST'] == '2022-12-22']
players_in_game = filtered_games_details['PLAYER_NAME'].unique()
players_df = pd.DataFrame(players_in_game, columns=['Player Name'])

# 合并两个数据集以获取球员薪水
players_salaries = pd.merge(players_df, nba_salaries, left_on='Player Name', right_on='Player Name', how='left')
players_salaries = players_salaries.dropna(subset=['Salary'])

# 筛选在2022-12-22有比赛的球员
nba_salaries_grouped = nba_salaries[nba_salaries['Player Name'].isin(players_salaries['Player Name'])]

# 定义模拟次数和每组球员数量
num_simulations = 10000
num_players_per_team = 5

# 进行模拟，随机选择球员并计算分数
simulation_results = []
simulation_player_names = []

for _ in range(num_simulations):
    selected_players = nba_salaries_grouped.sample(num_players_per_team, replace=False)  # 确保每次选择不重复球员
    team_score = selected_players['AverageSimulatedScore'].sum()
    simulation_results.append(team_score)
    simulation_player_names.append(selected_players[['Player Name', 'Position', 'Salary', 'AverageSimulatedScore']].to_dict(orient='records'))

# 绘制模拟结果的分布图
plt.figure(figsize=(12, 6))
sns.histplot(simulation_results, kde=True, bins=30, color='blue')
plt.xlabel('Sum of Average Simulated Fantasy Points')
plt.ylabel('Frequency')
plt.title('Distribution of Team Scores from Simulations')
plt.show()

# 定义模拟次数和每组球员数量
num_simulations = 10000
num_players_per_team = 5

# 进行模拟，随机选择球员并计算分数
simulation_results = []
simulation_player_names = []

for _ in range(num_simulations):
    selected_players = nba_salaries_grouped.sample(num_players_per_team, replace=False)  # 确保每次选择不重复球员
    team_score = selected_players['AverageSimulatedScore'].sum()
    team_salary = selected_players['Salary'].sum()
    sf_count = (selected_players['Position'] == 'SF').sum()
    unique_positions = selected_players['Position'].nunique()

    if team_score >=50 and team_salary <= 25000000 and sf_count >= 2 and unique_positions == 4:
        simulation_results.append((team_score, selected_players[['Player Name', 'Position', 'Salary', 'AverageSimulatedScore']].to_dict(orient='records')))

# 绘制模拟结果的分布图
if simulation_results:
    # 输出模拟结果的一些统计信息
    team_scores = [result[0] for result in simulation_results]
    print(f"Mean team score: {np.mean(team_scores)}")
    print(f"Standard deviation of team scores: {np.std(team_scores)}")
    print(f"Minimum team score: {np.min(team_scores)}")
    print(f"Maximum team score: {np.max(team_scores)}")

    # 找到符合条件的最大总分组合
    best_combination = max(simulation_results, key=lambda x: x[0])
    print(f"最佳组合的总分: {best_combination[0]}")
    print("球员详情:")
    for player in best_combination[1]:
        print(f"Player: {player['Player Name']}, Position: {player['Position']}, Salary: {player['Salary']}, PredictedFantasyPoints: {player['AverageSimulatedScore']}")

    # 获取最佳组合中球员的名称、预测分数和实际分数
    selected_player_names = [player['Player Name'] for player in best_combination[1]]
    predicted_scores = [player['AverageSimulatedScore'] for player in best_combination[1]]
    actual_scores = []

    for player_name in selected_player_names:
        actual_score = games_details[(games_details['PLAYER_NAME'] == player_name) & (games_details['GAME_DATE_EST'] == '2022-12-22')]['PTS'].values[0]
        actual_scores.append(actual_score)

    # 绘制条形图
    x = np.arange(len(selected_player_names))  # 球员数量
    width = 0.35  # 条形图的宽度

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, predicted_scores, width, label='Predicted Scores')
    bars2 = ax.bar(x + width/2, actual_scores, width, label='Actual Scores')

    # 添加标签和标题
    ax.set_xlabel('Player Names')
    ax.set_ylabel('Scores')
    ax.set_title('Predicted vs Actual Scores for Selected Players')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_player_names)
    ax.legend()

    # 在每个条形图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()
else:
    print("没有符合条件的组合。")

