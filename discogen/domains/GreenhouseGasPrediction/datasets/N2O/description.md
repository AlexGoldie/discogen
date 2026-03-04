DESCRIPTION
This dataset contains monthly nitrous oxide (N₂O) concentration measurements from NOAA's Global Monitoring Laboratory network, representing globally-averaged atmospheric levels from 2001 to 2014. Nitrous oxide is a long-lived greenhouse gas and ozone-depleting substance with sources including agricultural fertilizers, industrial processes, and natural soil emissions. The task is to forecast N₂O concentrations from 2015 to 2025.

OBSERVATION SPACE
Each observation is a time-stamped measurement with the following features:

Day of month: Approximated as 15 for all monthly measurements
Month: Integer from 1-12
Year: 2001-2014 (training period)
Day of year: Approximated from month
N₂O concentration: Parts per billion (ppb)

Data format: NumPy array of shape (N, 5) where N is the number of monthly observations.

TARGET SPACE
Continuous prediction of N₂O concentration in parts per billion (ppb) for future time points.

TEMPORAL STRUCTURE
Training period: January 2001 - December 2014 (14 years)
Evaluation period: January 2015 - December 2025 (11 years)
Sampling frequency: Monthly
Forecast horizon: Up to 132 months (11 years) into the future

EVALUATION METRICS
Models are evaluated using Mean Squared Error (MSE) on predictions for 2015-2025:

MSE = (1/N) × Σ(predicted - actual)²
Lower MSE indicates better forecasting performance

TASK OBJECTIVE
Develop a time series forecasting model that captures the long-term trend in atmospheric nitrous oxide and any seasonal or cyclical patterns. The model must extrapolate nearly as far into the future as the available training history spans, testing generalization beyond the observed regime.
