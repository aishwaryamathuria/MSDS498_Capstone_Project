# MSDS498_Capstone_Project
MSDS498_Capstone_Project

## Backend:
```
    cd dashboard/backend
    source .venv_mps_test311/bin/activate
    pip3.11 install -r dashboard/backend/requirements.txt
    python3.11 dashboard/backend/app.py
```
## Frontend:
```
    cd dashboard/frontend
    npm install
    npm run dev
```

## File/Folder Summary

### agents
- **hematology_agent.py**: Hematology report analyser using SLM + RAG
- **imaging_agent.py**: Chest X-Ray imaging analyser using fine tuned Densenet-121 and VLM
- **smol_tools.py**: smolagent framework for initializing smol agents for the above ones
- **validator_agent.py**: Evaluator using the conclusion of indivisual to set final claim status to ACCEPT/REJECT/UNCERTAIN

### dashboard
- **backend**: Backend APIs in Python Flask for claims dashboard to submit, view claims and update claim status
- **frontend**: Frontend React app for claims dashboard
- **data**: Store for submitted claims, their status and explanation

### model
- fine tuned Densenet 121 CV model

### model_exploration
- **Hematology_RAG_Evaluation.ipynb**: Exploration and evaluation of RAG performance and its metrics used in hematology agent
- **Imaging_CV_Model_Exploration.ipynb**: Exploration of multiple CV model to finalize the best model to use for pneumonia classification in out pipeline
- **VLM_Test_Explanations_Evaluation.ipynb**: Evaluation of x-ray human readable explanation generator VLM and its output quality
- **VLM_explanation.ipynb**: Exploration of x-ray human readable explanation generator VLM.
- **train_densenet121.py**: IPython notebook with training code for fine-tuning the CV nodel

### orchestrator.py
Agent orchestrator for invoking the required agents on claim submission.
