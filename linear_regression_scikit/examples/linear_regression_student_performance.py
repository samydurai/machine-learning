from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from linear_regression_basic_impl.linear_regression_dataset_parser import DataSetParser
from linear_regression_scikit.examples.plot_linear_regression_results import LinearRegressionResultPlot

dataset_slug = "nikhil7280/student-performance-multiple-linear-regression"
file_name_in_dataset = "Student_Performance.csv"
x_train_features = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours',
                    'Sample Question Papers Practiced']
y_train_label = 'Performance Index'
dataset_parser = DataSetParser()
x_train_df, y_train_df, df = dataset_parser.parse(dataset_slug, file_name_in_dataset, x_train_features, y_train_label)

numerical_features = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
categorical_features = ['Extracurricular Activities']

numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Create the full pipeline
# The pipeline first applies the preprocessor, then trains the regressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    x_train_df, y_train_df, test_size=0.2, random_state=42
)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

plot_results = LinearRegressionResultPlot()
plot_results.plot_residuals(y_test, y_pred)
plot_results.plot_residual_distribution(y_test, y_pred)
plot_results.plot_predictions_vs_actual(X_train, y_train, y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

linear_regression_model = model_pipeline.named_steps['regressor']

coefficients = linear_regression_model.coef_

intercept = linear_regression_model.intercept_

print(f'Coefficients {coefficients}')
print(f'Intercept {intercept}')
print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
