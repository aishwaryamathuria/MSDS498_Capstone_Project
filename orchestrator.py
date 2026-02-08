from agents.imaging_agent import run as imaging_agent_run
from agents.hematology_agent import run as hematology_agent_run


def run_patient_workflow(image_path=None, hematology_report_path=None):
    """
    Trigger the right agent(s) based on input and return their interpretation summaries.
    - If image_path is provided, triggers the imaging agent and gets its summary.
    - If hematology_report_path is provided, triggers the hematology agent and gets its summary.
    The orchestrator only routes to agents and returns what they return; it does not interpret.
    """
    results = []
    if image_path:
        results.append(imaging_agent_run(image_path=image_path))
    if hematology_report_path:
        results.append(hematology_agent_run(report_path=hematology_report_path))
    if not results:
        return "No inputs provided. Give image_path and/or hematology_report_path."
    return "\n\n".join(results)


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--code-agent"]
    image_path = args[0] if len(args) > 0 else "test_cases/Patient 1/IM-0001-0001.jpeg"
    hema_path = args[1] if len(args) > 1 else "positive/patient1.txt"
    print("Orchestrator: triggering agents based on input...\n")
    result = run_patient_workflow(
        image_path=image_path,
        hematology_report_path=hema_path,
    )
    print("--- Result (agent summaries) ---\n")
    print(result)
    print("\n\n")
