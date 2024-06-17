import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('/Users/hy/Documents/Y2S2/Management_Science/MS_FinalProject/games_details.csv')

# Convert 'MIN' column to total minutes
def convert_minute_format(minute_str):
    if isinstance(minute_str, str):
        parts = minute_str.split(':')
        if len(parts) == 3:  # 'HH:MM:SS' format
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
        elif len(parts) == 2:  # 'MM:SS' format
            return int(parts[0]) + int(parts[1]) / 60
    return 0  # Default to 0 if not a valid format

df['MIN'] = df['MIN'].apply(convert_minute_format)

# Define the fantasy score formula
df['FANTASY_SCORE'] = (
    df['PTS'] +
    1.2 * df['REB'] +
    1.5 * df['AST'] +
    3 * df['STL'] +
    3 * df['BLK'] -
    df['TO']
)

# Create one-hot encoded features for 'START_POSITION' and 'TEAM_ABBREVIATION'
df = pd.get_dummies(df, columns=['START_POSITION', 'TEAM_ABBREVIATION'], drop_first=True)

# Select features and target variable
features = ['MIN', 'PLUS_MINUS'] + [col for col in df.columns if col.startswith('START_POSITION_') or col.startswith('TEAM_ABBREVIATION_')]
X = df[features]
y = df['FANTASY_SCORE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train R-squared: {train_r2}, Test R-squared: {test_r2}")
