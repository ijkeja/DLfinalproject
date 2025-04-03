This folder contains two notebooks, "JetClass_JetTagging" and "JetClass_evaluationMetrics".
The JetTagging file is responsible for importing, configuring and training our model while the evaluationMetrics file contains the performance analysis.
It is unfortunately necessary to separate this into two files since we were unable to resolve the numpy dependency issues within the same file. 
We train the model in the JetTagging file and then store the outputs in the files "training_history.json", "y_probs.json" and "y_true.json".
These outputs are then used to evaluate the model in the evaluationMetrics file.
