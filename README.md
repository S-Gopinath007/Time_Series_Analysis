# Time_Series_Analysis
1.Optimized Python code implementation for data preparation, model training. tuning, and evaluation.
2.Final Report: Optimized Multivariate LSTM for Stock Price Forecasting

Hyperparameter Search Space and Results:
Hyperparameter tuning was conducted using GridSearchCV combined with {KerasRegressor} on the combined training and validation sets text{train\_val} to find the best configuration that minimizes the Mean Squared Error (MSE).
Search Space
The search was constrained to effectively explore the most critical architecture elements: complexity (units), regularization (dropout), and learning speed (learning rate).
Hyperparameter,Description,Search Values
Units per Layer,The number of memory units in the LSTM layers.,"{50,60}"
Dropout Rate,The percentage of neurons randomly dropped for regularization.,"{0.1,0.3}"
Learning Rate,The step size for the Adam optimizer.,"{0.001,0.0005}"
Batch Size,Number of samples processed before the model update.,"{16,32}"

Final Optimized Configuration:
The best model configuration identified by $\text{GridSearchCV}$ yielded the highest performance. The architecture consisted of two stacked LSTM layers.
Parameter,Value
Architecture,2 Stacked LSTM Layers
Units per Layer,60
Dropout Rate,0.1
Learning Rate,0.001
Batch Size,32

Justification for Optimized Configuration:
The final choice of 60 units and a low 0.1 dropout rate suggests that the complex dynamics of the multivariate stock data benefit from a richer model capacity (more units) combined with a less aggressive regularization (lower dropout) to effectively learn the patterns within the limited 10-day sequence window. This configuration led to the lowest prediction error metrics

3. Comparative analysis text summarizing the performance metrics (RMSE, MAE) of the Baseline LSTM, Optimized LSTM, and Classical Benchmark
Model,RMSE (Root Mean Squared Error),MAE (Mean Absolute Error)
Optimized LSTM,$1.3659,$0.9583
"ARIMA(5,1,0) Benchmark",$1.3965,$1.0505
Baseline LSTM,$1.8122,$1.3655

summary:
Best Performance: The Optimized LSTM achieved the lowest error across both metrics, validating the effectiveness of tuning the deep learning architecture.
Significant Improvement: The optimization process led to a 25% reduction in RMSE and a 30% reduction in MAE compared to the Baseline LSTM.
Benchmark Win: The Optimized LSTM narrowly but consistently beat the highly competitive ARIMA(5,1,0) model, demonstrating the marginal but critical advantage of modeling non-linear, multivariate relationships in financial data.

4.Written analysis explaining the findings from the model explainability technique, specifically highlighting which historical features or time steps the model relied upon most heavily for its final predictions
The Optimized LSTM model relies most heavily on the most recent trading days within the input window, and its influence drops off rapidly with time.Findings from ExplainabilityPriority on Recent Data: The three most recent time steps (t-1,t-2,t-3) collectively account for over 37% of the model's total predictive influence.Decay: The influence of past data shows a clear monotonic decay. The most recent day (t-1) has the highest influence (0.1373), while the oldest day (t-10) has the lowest ($\mathbf{0.0573}$).Conclusion: The model correctly learned that short-term momentum is the most critical factor for accurate stock price forecasting, validating the choice of the 10-day lookback window.
