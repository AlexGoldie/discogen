DESCRIPTION
This dataset contains daily carbon dioxide (CO₂) concentration measurements from the Mauna Loa Observatory in Hawaii, spanning May 1974 to December 2019. Located at 3,200 meters above sea level on an isolated mid-Pacific volcano, Mauna Loa has provided the world's longest continuous record of atmospheric CO₂ since 1958, making it the definitive benchmark for tracking global climate change. The task is to forecast daily CO₂ levels from 2020 to 2025.

OBSERVATION SPACE
Each observation is a daily measurement with the following features:

Day of month: Integer from 1-31
Month: Integer from 1-12
Year: 1974-2019 (training period)
Day of year: Integer from 1-366
CO₂ concentration: Parts per million (ppm)

Data format: NumPy array of shape (N, 5) where N is the number of daily observations.

TARGET SPACE
Continuous prediction of CO₂ concentration in parts per million (ppm) for future dates.

TEMPORAL STRUCTURE
Training period: May 1974 - December 2019 (45 years, days with missing measurements omitted)
Evaluation period: January 2020 - November 2025 (6 years)
Sampling frequency: Daily
Forecast horizon: Up to ~2,160 days (6 years) into the future

EVALUATION METRICS
Models are evaluated using Mean Squared Error (MSE) on daily predictions for 2020-2025:

MSE = (1/N) × Σ(predicted - actual)²
Lower MSE indicates better forecasting accuracy

TASK OBJECTIVE
Develop a time series forecasting model that accurately captures both the secular trend in atmospheric CO₂ and the regular seasonal oscillations. The model must extrapolate 6 years beyond training while maintaining daily-level precision, requiring understanding of both long-term forcing and periodic biological cycles.
