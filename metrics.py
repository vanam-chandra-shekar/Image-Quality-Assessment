import tensorflow as tf

from cnn_model import X_test

# If you saved it as .h5
model = tf.keras.models.load_model("model.h5")

# OR if you saved it using the SavedModel format
# model = tf.keras.models.load_model("path/to/saved_model")
y_pred = model.predict(X_test).flatten()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np

# Ensure ground truth is a NumPy array
y_true = y_test.flatten()

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

plcc, _ = pearsonr(y_true, y_pred)
srocc, _ = spearmanr(y_true, y_pred)
krocc, _ = kendalltau(y_true, y_pred)

print(f"📊 MAE:   {mae:.4f}")
print(f"📊 RMSE:  {rmse:.4f}")
print(f"📊 R²:    {r2:.4f}")
print(f"📊 PLCC:  {plcc:.4f}")
print(f"📊 SROCC: {srocc:.4f}")
print(f"📊 KROCC: {krocc:.4f}")
