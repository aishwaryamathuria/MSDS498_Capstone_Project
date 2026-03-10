# MSDS498_Capstone_Project
## An Agentic AI Framework For Medical Imaging And Report Analysis For Health Insurance Claim Validation

The proposed system is designed as an AI-assisted decision-support architecture that helps medical reviewers during the health insurance claim adjudication process. In real-world insurance workflows, adjudication is a complex and high-volume task that significantly affects cost, efficiency, and decision consistency. The architecture therefore focuses on supporting human judgment by pre-screening medical evidence and highlighting cases where the clinical claim may not be well supported by imaging or report data.

The core of the system is a modular pipeline built around specialized AI components. A computer vision model first analyzes chest X-ray images to identify visual patterns that are commonly associated with pneumonia. Instead of producing a final decision, this component generates structured outputs such as confidence scores and detected imaging features. These outputs are designed to be interpretable so they can be reviewed and used as evidence in case of uncertain submissions rather than treated as automated conclusions.

The outputs from the imaging model are then passed to a vision-language reasoning component that generates short, human-readable explanations. These explanations describe why the available imaging evidence supports or weakly supports the claimed diagnosis, with an initial focus on rejected or questionable claims. The explanations are intentionally constrained to reflect only model findings and established clinical concepts, helping maintain clinical grounding and transparency. Another hematology agent evaluates the claim validity based on the provided report and generates a human readable explanation accepting or refuting the claim. All results are then evaluated to check concensus or disagreement to make the final say or forward for expert review.


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
