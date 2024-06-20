import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# 讀取數據
df = pd.read_csv('games_details.csv')

# 將時間轉換為數值（分鐘數）
def convert_min_to_float(min_str):
    try:
        if isinstance(min_str, str) and ':' in min_str:
            parts = min_str.split(':')
            return int(parts[0]) + int(parts[1]) / 60
        elif isinstance(min_str, (int, float)):
            return float(min_str)
    except ValueError:
        return 0
    return 0

df['MIN'] = df['MIN'].apply(convert_min_to_float)

# 計算初始夢幻分數
df['fantasy_points'] = (df['PTS'] * 1) + (df['REB'] * 0.5) + (df['AST'] * 1.2) + (df['STL'] * 1) + (df['BLK'] * 1.2) - (df['TO'] * 1) + (df['FG3M'] * 0.5)

# 填補 fantasy_points 列中的 NaN 值
df['fantasy_points'] = df['fantasy_points'].fillna(0)

# 選擇特徵欄位
features = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS']
X = df[features]
y = df['fantasy_points']

# 處理缺失值
X = X.fillna(0)

# 使用 LassoCV 自動選擇最佳的正則化強度
lasso = LassoCV(cv=5, random_state=0).fit(X, y)

# 使用 SelectFromModel 選擇重要特徵
model = SelectFromModel(lasso, prefit=True)
X_selected = model.transform(X)

# 選中的特徵名
selected_features = np.array(features)[model.get_support()]

print("Selected features using Lasso:", selected_features)

# 重新計算夢幻分數，僅使用選定的特徵
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
lasso_model = lasso.fit(X_selected_df, y)
y_pred = lasso_model.predict(X_selected_df)

# 評估模型準確性
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)


# 繪製實際值與預測值的比較
plt.scatter(y, y_pred)
plt.xlabel("Actual Fantasy Points")
plt.ylabel("Predicted Fantasy Points")
plt.title("Actual vs Predicted Fantasy Points")
plt.show()