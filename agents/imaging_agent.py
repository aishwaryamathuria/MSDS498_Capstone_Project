def dummy_imaging_analysis(image_path=None, **kwargs):
    return {
        "triggered": True,
        "image_path": image_path or "none",
        "status": "dummy_analysis_complete",
        "mock_findings": "No actual analysis performed (dummy).",
    }


def run(image_path=None):
    """Run imaging agent and return a short interpretation summary."""
    result = dummy_imaging_analysis(image_path=image_path)
    return (
        f"Imaging: {result['status']}. "
        f"Image: {result['image_path']}. {result['mock_findings']}"
    )
