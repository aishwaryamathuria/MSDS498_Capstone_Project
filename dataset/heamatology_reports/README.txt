Golden dataset for hematology RAG classification.

- positive/  Lab reports labeled as pneumonia (positive). Used as positive examples for the agent.
- negative/  Lab reports labeled as no pneumonia (negative). Used as negative examples for the agent.

The agent loads a sample of these reports and passes them to the language model as few-shot examples when classifying a new report.
