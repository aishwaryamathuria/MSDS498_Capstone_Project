"""
Agent Orchestrator
------------------
Routes requests to appropriate agents based on availability
of imaging and/or hematology data in the report.
"""

from imaging_agent import ImagingAgent
from hematology_agent import HematologyAgent


class AgentOrchestrator:
    def __init__(self):
        # Initialize agents
        self.imaging_agent = ImagingAgent()
        self.hematology_agent = HematologyAgent()

    def route_based_on_report(self, report: dict):
        """
        Routes to agents based ONLY on what is present in the report.

        Expected report format example:
        {
            "imaging": {...} or None,
            "hematology": {...} or None
        }
        """

        print("[Orchestrator] Evaluating report availability...")

        responses = {}

        # Check availability
        has_imaging = report.get("imaging") is not None
        has_hematology = report.get("hematology") is not None

        # Case 1: Only Imaging available
        if has_imaging and not has_hematology:
            print("[Orchestrator] -> Invoking Imaging Agent only")
            responses["imaging_result"] = self.imaging_agent.process(
                report["imaging"]
            )

        # Case 2: Only Hematology available
        elif has_hematology and not has_imaging:
            print("[Orchestrator] -> Invoking Hematology Agent only")
            responses["hematology_result"] = self.hematology_agent.process(
                report["hematology"]
            )

        # Case 3: Both Imaging and Hematology available
        elif has_imaging and has_hematology:
            print("[Orchestrator] -> Invoking BOTH Imaging and Hematology Agents")

            responses["imaging_result"] = self.imaging_agent.process(
                report["imaging"]
            )

            responses["hematology_result"] = self.hematology_agent.process(
                report["hematology"]
            )

        # Case 4: Nothing available
        else:
            return {
                "status": "error",
                "message": "No valid imaging or hematology data found in report"
            }

        return {
            "orchestrator_status": "completed",
            "invoked_agents": responses
        }
