import xgboost
import shap

# train an XGBoost model
X, y = shap.datasets.california()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize
shap.plots.waterfall(shap_values[0])
shap.plots.force(shap_values[0])
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values[0])