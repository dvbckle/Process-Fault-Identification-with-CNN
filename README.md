# Process-Fault-Identification-with-CNN
This repo has a CNN model created to identify faults in a simulated process dataset.  Recently I added a second notebook using the simulated dataset but also adding rate of change of process control values and a second derivative of those values. I plan to add Notebooks using this methodology to predict sheet breaks in a Paper Machine dataset.  

Currently there are these notebook:
- One with the simulated dataset using only the process control values.
- One with the simulated dataset adding 'velocity' & 'acceleration' of the process control values.
- One with EDA on the paper machine dataset.

# Business Case
Identifying an impending process interuption, fault or failure in advance to warn operators to take remedial action, or program the process control system to take a different action than maintaining readings within set point ranges may reduce the number of process interuptions, thus improving production and reducing costs.

# Assumptions
Process variation within set point limits or specific or correlated deviation by a subset of measurements may provide advance indication of an upcoming failure.
Some, but not all sheet breaks may occur due to rapid deviation in a control point or an external issue such as a power failure, auxiliary equipment failure, or human intervention - some type of emergency shut down.
Other sheet breaks may result from a combined effect of two or several process measurements where the deviation of each is within limits, but the combined effect leads to a sheet break. Otherwise the proces control if running within set limits should prevent a sheet break. If the assumption of combined deviation does not hold, then most sheet breaks are a result of some variation of auxhiliary equipment mal-function that the process control system cannot correct.
