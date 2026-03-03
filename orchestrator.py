# from agents.imaging_agent import analyze_imaging
# from agents.hematology_agent import analyze as analyze_hematology


# # def _imaging_probability_decision(probability):
# #     if probability > 0.95:
# #         return "accept"
# #     if probability < 0.7:
# #         return "reject"
# #     return "uncertain"
# def _imaging_band_decision(band_label):
#     band_label = (band_label or "").upper()

#     if band_label == "PNEUMONIA":
#         return "accept"
#     if band_label == "NORMAL":
#         return "reject"
#     return "uncertain"


# # Trigger selected agents and return structured outputs + final claim status.
# def run_patient_workflow(image_path=None, hematology_report_path=None):
#     agent_results = []
#     confident_imaging_decisions = []

#     if image_path:
#         imaging_result = analyze_imaging(image_path=image_path)

#         probability = imaging_result.get("probability")
#         decile = imaging_result.get("decile")
#         band_label = imaging_result.get("band_label", "UNCERTAIN")
#         explanation = imaging_result.get("explanation", "No imaging explanation generated.")

#         decision = _imaging_band_decision(band_label)

#         if decision in {"accept", "reject"}:
#             confident_imaging_decisions.append(decision)

#         agent_results.append(
#             {
#                 "agent": "imaging",
#                 "decision": decision,
#                 "probability": float(probability) if isinstance(probability, (int, float)) else None,
#                 "decile": decile,
#                 "band_label": band_label,
#                 "explanation": explanation,
#             }
#         )

#     if hematology_report_path:
#         hema_result = analyze_hematology(report_path=hematology_report_path)
#         agent_results.append(
#             {
#                 "agent": "hematology",
#                 "decision": hema_result.get("decision", "uncertain"),
#                 "probability": None,
#                 "explanation": hema_result.get(
#                     "explanation", "Hematology: No interpretation generated."
#                 ),
#             }
#         )

#     if not agent_results:
#         return {
#             "status": "uncertain",
#             "agent_results": [],
#             "message": "No inputs provided. Give image_path and/or hematology_report_path.",
#         }

#     decisions = [item["decision"] for item in agent_results]
#     all_accept = all(item == "accept" for item in decisions)
#     all_reject = all(item == "reject" for item in decisions)

#     # Rule requested:
#     # - all invoked agents accept + all present probabilities > 0.9 -> accept
#     # - all invoked agents reject + all present probabilities < 0.7 -> reject
#     # - otherwise uncertain
#     # has_probability = len(probabilities) > 0
#     # if all_accept and has_probability and all(prob > 0.9 for prob in probabilities):
#     #     final_status = "accept"
#     # elif all_reject and has_probability and all(prob < 0.7 for prob in probabilities):
#     #     final_status = "reject"
#     # else:
#     #     final_status = "uncertain"

#     # return {"status": final_status, "agent_results": agent_results}
#     if all_accept:
#         final_status = "accept"
#     elif all_reject:
#         final_status = "reject"
#     else:
#         final_status = "uncertain"

#     return {"status": final_status, "agent_results": agent_results}


# if __name__ == "__main__":
#     import sys
#     args = [a for a in sys.argv[1:] if a != "--code-agent"]
#     image_path = args[0] if len(args) > 0 else "test_cases/Patient 1/IM-0001-0001.jpeg"
#     hema_path = args[1] if len(args) > 1 else "positive/patient1.txt"
#     print("Orchestrator: triggering agents based on input...\n")
#     result = run_patient_workflow(
#         image_path=image_path,
#         hematology_report_path=hema_path,
#     )
#     print("--- Result (structured) ---\n")
#     print(f"Final status: {result['status']}\n")
#     for item in result["agent_results"]:
#         print(f"[{item['agent']}] {item['explanation']}\n")
#     print("\n\n")


from agents.imaging_agent import analyze_imaging
from agents.hematology_agent import analyze as analyze_hematology
from agents.validator_agent import validate_claim


def _imaging_band_decision(band_label):
    band_label = (band_label or "").upper()

    if band_label == "PNEUMONIA":
        return "accept"
    if band_label == "NORMAL":
        return "reject"
    return "uncertain"


# Trigger selected agents and return structured outputs + final claim status.
def run_patient_workflow(image_path=None, hematology_report_path=None):
    agent_results = []
    imaging_payload = None
    hematology_payload = None

    if image_path:
        imaging_result = analyze_imaging(image_path=image_path)

        probability = imaging_result.get("probability")
        decile = imaging_result.get("decile")
        band_label = imaging_result.get("band_label", "UNCERTAIN")
        explanation = imaging_result.get("explanation", "No imaging explanation generated.")

        decision = _imaging_band_decision(band_label)

        imaging_payload = {
            "decision": decision,
            "probability": float(probability) if isinstance(probability, (int, float)) else None,
            "decile": decile,
            "band_label": band_label,
            "explanation": explanation,
        }

        agent_results.append(
            {
                "agent": "imaging",
                **imaging_payload,
            }
        )

    if hematology_report_path:
        hema_result = analyze_hematology(report_path=hematology_report_path)

        hematology_payload = {
            "decision": hema_result.get("decision", "uncertain"),
            "verdict": hema_result.get("verdict"),
            "values": hema_result.get("values"),
            "probability": None,
            "decile": None,
            "band_label": None,
            "explanation": hema_result.get(
                "explanation", "Hematology: No interpretation generated."
            ),
        }

        agent_results.append(
            {
                "agent": "hematology",
                **hematology_payload,
            }
        )

    if not agent_results:
        return {
            "status": "uncertain",
            "agent_results": [],
            "message": "No inputs provided. Give image_path and/or hematology_report_path.",
        }

    validator_result = validate_claim(
        imaging_result=imaging_payload,
        hematology_result=hematology_payload,
    )

    return {
        "status": validator_result["decision"],
        "agent_results": agent_results,
        "validator_result": validator_result,
    }


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

    print("--- Result (structured) ---\n")
    print(f"Final status: {result['status']}\n")

    for item in result["agent_results"]:
        print(f"[{item['agent']}] decision={item['decision']}")
        print(f"Explanation: {item['explanation']}\n")

    if "validator_result" in result:
        print("[validator]")
        print(f"decision={result['validator_result']['decision']}")
        print(f"rule={result['validator_result']['decision_rule']}")
        print(f"source={result['validator_result'].get('source')}")
        print(f"Explanation: {result['validator_result']['explanation']}\n")
