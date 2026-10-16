
# Optional: retrain models (run only if you want to reproduce training)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib

train_days = 20
df_train = df[df['day'] < train_days].copy()
df_test2 = df[df['day'] >= train_days].copy()

features = ['G', 'T_cell', 'P_emp', 'hour', 'dayofyear']

X_train = df_train[features].fillna(0.0)
y_train = df_train['P_obs']

X_test2 = df_test2[features].fillna(0.0)
y_test2 = df_test2['P_obs']

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

gbr = GradientBoostingRegressor(
    n_estimators=250,
    max_depth=4,
    random_state=0
)
gbr.fit(X_train, y_train)

print('Ridge MAE:', mean_absolute_error(y_test2, ridge.predict(X_test2)))
print('GBR  MAE:', mean_absolute_error(y_test2, gbr.predict(X_test2)))

joblib.dump(gbr, 'gbr_model.joblib')
joblib.dump(ridge, 'ridge_model.joblib')

print('Models saved.')
