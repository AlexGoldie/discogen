DESCRIPTION
This dataset contains daily carbon dioxide (CO₂) concentration measurements from the Mauna Loa Observatory in Hawaii, spanning 1970 to 2010. Located at 3,200 meters above sea level on an isolated mid-Pacific volcano, Mauna Loa has provided the world's longest continuous record of atmospheric CO₂ since 1958, making it the definitive benchmark for tracking global climate change. The task is to forecast daily CO₂ levels from 2010 to 2025.

OBSERVATION SPACE
Each observation is a daily measurement with the following features:

Year: 1970-2010 (training period)
Month: Integer from 1-12
Day: Integer from 1-31
Fractional year: Precise decimal representation (e.g., 1974.3781)
CO₂ concentration: Parts per million (ppm)

Data format: NumPy array of shape (N, 5) where N is the number of daily observations.

TARGET SPACE
Continuous prediction of CO₂ concentration in parts per million (ppm) for future dates.

TEMPORAL STRUCTURE
Training period: 1970-2010 (40 years)
Evaluation period: 2010-2025 (15 years)
Sampling frequency: Daily
Forecast horizon: Up to ~5,475 days (15 years) into the future

EVALUATION METRICS
Models are evaluated using Mean Squared Error (MSE) on daily predictions for 2010-2025:

MSE = (1/N) × Σ(predicted - actual)²
Lower MSE indicates better forecasting accuracy

TASK OBJECTIVE
Develop a time series forecasting model that accurately captures both the secular trend in atmospheric CO₂ and the regular seasonal oscillations. The model must extrapolate 15 years beyond training while maintaining daily-level precision, requiring understanding of both long-term forcing and periodic biological cycles.
