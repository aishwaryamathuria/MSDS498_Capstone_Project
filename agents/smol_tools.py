from typing import Optional

from smolagents import tool

from agents.imaging_agent import run as imaging_agent_run
from agents.hematology_agent import run as hematology_agent_run


def run_imaging_analysis_impl(image_path: Optional[str] = None) -> str:
    return imaging_agent_run(image_path=image_path)


@tool
def run_imaging_analysis(image_path: Optional[str] = None) -> str:
    """Trigger imaging analysis (e.g. chest X-ray). Returns status and mock findings.
    Args:
        image_path: Path to the image file, or None for a dummy run.
    """
    return run_imaging_analysis_impl(image_path)

def check_hematology_report_impl(report_path: str) -> str:
    return hematology_agent_run(report_path=report_path)


@tool
def check_hematology_report(report_path: str) -> str:
    """Check a hematology report for pneumonia markers. Returns verdict (true/false/uncertain) and values.
    Args:
        report_path: Path to the report file, e.g. positive/patient1.txt or negative/patient1.txt.
    """
    return check_hematology_report_impl(report_path)
