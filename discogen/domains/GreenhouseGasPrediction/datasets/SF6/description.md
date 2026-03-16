DESCRIPTION
This dataset contains monthly sulfur hexafluoride (SF₆) concentration measurements from NOAA's Global Monitoring Laboratory network, representing globally-averaged atmospheric levels from 1997 to 2014. SF₆ is an entirely synthetic compound used as an electrical insulator in high-voltage equipment and has the highest global warming potential of any known substance. Its presence in the atmosphere is purely anthropogenic, making it a unique tracer of industrial activity. The task is to forecast SF₆ concentrations from 2015 to 2025.

OBSERVATION SPACE
Each observation is a time-stamped measurement with the following features:

Day of month: Approximated as 15 for all monthly measurements
Month: Integer from 1-12
Year: 1997-2014 (training period)
Day of year: Approximated from month
SF₆ concentration: Parts per trillion (ppt)

Data format: NumPy array of shape (N, 5) where N is the number of monthly observations.

TARGET SPACE
Continuous prediction of SF₆ concentration in parts per trillion (ppt) for future time points.

TEMPORAL STRUCTURE
Training period: January 1997 - December 2014 (18 years)
Evaluation period: January 2015 - December 2025 (11 years)
Sampling frequency: Monthly
Forecast horizon: Up to 132 months (11 years) into the future

EVALUATION METRICS
Models are evaluated using Mean Squared Error (MSE) on predictions for 2015-2025:

MSE = (1/N) × Σ(predicted - actual)²
Lower MSE indicates better forecasting performance

TASK OBJECTIVE
Develop a time series forecasting model that captures the long-term trend in atmospheric sulfur hexafluoride. Unlike other greenhouse gases, SF₆ exhibits primarily secular growth with minimal seasonal modulation, making trend extrapolation the primary challenge. The model must forecast 11 years into the future based on 18 years of historical data.
