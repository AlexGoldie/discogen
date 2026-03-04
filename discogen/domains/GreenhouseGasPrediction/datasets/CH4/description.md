DESCRIPTION
This dataset contains monthly methane (CH₄) concentration measurements from NOAA's Global Monitoring Laboratory network, representing globally-averaged atmospheric levels from 1983 to 2014. Methane is a potent greenhouse gas with both natural and anthropogenic sources, including wetlands, agriculture, fossil fuel extraction, and waste decomposition. The task is to forecast future CH₄ concentrations from 2015 to 2025 based on historical patterns.

OBSERVATION SPACE
Each observation is a time-stamped measurement with the following features:

Day of month: Approximated as 15 for all monthly measurements
Month: Integer from 1-12
Year: 1983-2014 (training period)
Day of year: Approximated from month (e.g., January ≈ 15, February ≈ 46)
CH₄ concentration: Parts per billion (ppb)

Data format: NumPy array of shape (N, 5) where N is the number of monthly observations.

TARGET SPACE
Continuous prediction of CH₄ concentration in parts per billion (ppb) for future time points.

TEMPORAL STRUCTURE
Training period: January 1983 - December 2014 (32 years)
Evaluation period: January 2015 - December 2025 (11 years)
Sampling frequency: Monthly
Forecast horizon: Up to 132 months (11 years) into the future

EVALUATION METRICS
Models are evaluated using Mean Squared Error (MSE) on predictions for 2015-2025:

MSE = (1/N) × Σ(predicted - actual)²
Lower MSE indicates better forecasting performance
Penalizes large errors more heavily than small ones

TASK OBJECTIVE
Develop a time series forecasting model that captures both the long-term upward trend in atmospheric methane and any seasonal or cyclical patterns present in the data. The model must extrapolate 11 years beyond the training period while maintaining accuracy.
