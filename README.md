# Process-Fault-Identification-with-CNN
UPDATE:  This repo is updated Feb 2022 with a new Notebook using the Paper Machine daaset. This notebook uses a slightly different CNN model, has some data corrections (apparent mislabelled samples) and a new chart methodology to present the results.  Notebook:  Paper_Machine_Sheet_Break_Subset96-3Class_w_Dates_Adamax.
I have included a summary discussion of using CNN's on the paper machine dataset.

This repo has a CNN models created to predict / warn of upcoming paper machine faults prior to the condition. Work in this repo began with a simulated process dataset and  CNN models to predict failures. A second simulated dataset notebook adds rate of change of process control values and a second derivative of those values.   
Paper Machine dataset source is also listed in the notebooks.

Reference and Licence
Dataset: Rare Event Classification in Multivariate Time Series
2018-10-01 Chitta Ranjan, Markku Mustonen, Kamran Paynabar, Karim Pourak
https://deeplearn.org/arxiv/48599/dataset:-rare-event-classification-in-multivariate-time-series

The dataset license details are shown in the file processminer-rare-event-mts - LICENSE.csv included from the data download site:
"- users can share the dataset or any publication that uses the data by giving credit to the data provider. Please cite this paper, https://arxiv.org/abs/1809.10717, for the credit."
- the dataset cannot be used for any commercial purposes.
- users can distribute any additions, transformations, or changes to your dataset under this license. However, the same license needs to be added to any redistributed data." Hence, any user of the adapted dataset would likewise need to share their work with this license.
Therefore, my work using this data is also shared under the same conditions as the data license. One cannot use the dataset or derivatives of the data for a commercial purpose.

Use or application of methods included here are at the sole risk of the reader. The author takes no responsibility for application by the reader.

https://arxiv.org/abs/1809.10717

Currently, notebooks included are:
- One with the simulated dataset using only the process control values.
- One with the simulated dataset adding 'velocity' & 'acceleration' of the process control values and five warning classes plus normal and faliure classesplus normal and faliure classes.
- One with the simulated dataset adding 'velocity' & 'acceleration' of the process control values and one warning class plus normal and failure classes
- One with EDA on the paper machine dataset.
- Five models with a subset of the paper machine data using the most frequent categorical feature subset.
      - One uses 5 warning classes
      - One uses a single warning class and performs better than the 5 warning class model.
      - One (updated) with a single warning class using a different CNN optimizer and a better presentation graphs
      - One with features derived from feature importance using a RF model with descrete samples with positio, velocity and acceleration - the best model
      - One with features derived from permutation importance using a RF model with descrete samples with positio, velocity and accelerationto 
      - The feature / permutation importance notebook used to create feature lists for the above tewo models
      - A comparison plot notebook showing difference between the custom classifier using predict_proba and the standard argmax result

# Business Case
Identifying an impending process interuption, fault or failure in advance to warn operators to take remedial action, or program the process control system to take a different action than maintaining readings within set point ranges may reduce the number of process interuptions, thus improving production and reducing costs.

# Assumptions
Process variation within set point limits or specific or correlated deviation by a subset of measurements may provide advance indication of an upcoming failure.
Some, but not all sheet breaks may occur due to rapid deviation in a control point or an external issue such as a power failure, auxiliary equipment failure, or human intervention - some type of emergency shut down.
Other sheet breaks may result from a combined effect of two or several process measurements where the deviation of each is within limits, but the combined effect leads to a sheet break. Otherwise the proces control if running within set limits should prevent a sheet break. If the assumption of combined deviation does not hold, then most sheet breaks are a result of some variation of auxhiliary equipment mal-function that the process control system cannot correct.
