from app.models.schemas import LabResult, MedicalReport, ResultFlag

# Mapping of common lab tests to organ-system panels
_PANELS: dict[str, list[str]] = {
    "Liver Panel": [
        "sgpt", "alt", "sgot", "ast", "alp", "alkaline phosphatase",
        "bilirubin", "total bilirubin", "direct bilirubin", "indirect bilirubin",
        "ggt", "gamma gt", "albumin", "globulin", "total protein",
    ],
    "Kidney Panel": [
        "creatinine", "urea", "bun", "blood urea nitrogen",
        "uric acid", "egfr", "gfr",
    ],
    "Lipid Panel": [
        "cholesterol", "total cholesterol", "hdl", "ldl", "vldl",
        "triglycerides", "triglyceride",
    ],
    "CBC (Complete Blood Count)": [
        "hemoglobin", "hb", "hgb", "rbc", "wbc", "platelet", "platelets",
        "plt", "hematocrit", "hct", "mcv", "mch", "mchc", "rdw",
        "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil",
    ],
    "Thyroid Panel": [
        "tsh", "t3", "t4", "ft3", "ft4", "free t3", "free t4",
    ],
    "Diabetes Panel": [
        "glucose", "fasting glucose", "blood sugar", "fbs", "ppbs",
        "hba1c", "glycated hemoglobin", "random blood sugar", "rbs",
    ],
}


def categorize_by_panel(report: MedicalReport) -> dict[str, list[LabResult]]:
    """Group lab results by organ-system panel."""
    panels: dict[str, list[LabResult]] = {}
    uncategorized: list[LabResult] = []

    for result in report.results:
        test_lower = result.test_name.lower().strip()
        placed = False
        for panel_name, keywords in _PANELS.items():
            if any(kw in test_lower for kw in keywords):
                panels.setdefault(panel_name, []).append(result)
                placed = True
                break
        if not placed:
            uncategorized.append(result)

    if uncategorized:
        panels["Other"] = uncategorized

    return panels


def get_abnormal_results(report: MedicalReport) -> list[LabResult]:
    """Return only the results flagged outside normal range."""
    return [r for r in report.results if r.flag != ResultFlag.NORMAL]


def build_structured_summary(report: MedicalReport) -> dict:
    """Produce a structured summary dict suitable for API responses or LLM context."""
    panels = categorize_by_panel(report)
    abnormal = get_abnormal_results(report)

    summary: dict = {
        "patient": report.patient.model_dump(),
        "report_type": report.report_type.value,
        "report_date": report.report_date,
        "total_tests": len(report.results),
        "normal_count": len(report.results) - len(abnormal),
        "abnormal_count": len(abnormal),
        "abnormal_results": [r.model_dump() for r in abnormal],
        "panels": {},
    }

    for panel_name, results in panels.items():
        panel_abnormal = [r for r in results if r.flag != ResultFlag.NORMAL]
        summary["panels"][panel_name] = {
            "total": len(results),
            "abnormal": len(panel_abnormal),
            "results": [r.model_dump() for r in results],
        }

    return summary


def build_llm_context(report: MedicalReport, max_results: int = 200) -> str:
    """Serialize a MedicalReport into a formatted text block for LLM consumption."""
    lines: list[str] = []

    # Patient header
    p = report.patient
    patient_parts = [f"Patient: {p.name}"]
    if p.age is not None:
        patient_parts.append(f"Age: {p.age}")
    if p.gender:
        patient_parts.append(f"Gender: {p.gender}")
    if p.patient_id:
        patient_parts.append(f"ID: {p.patient_id}")
    lines.append(" | ".join(patient_parts))
    lines.append(f"Report type: {report.report_type.value}")
    if report.report_date:
        lines.append(f"Date: {report.report_date}")
    lines.append("")

    # Tabular results grouped by panel
    if report.results:
        panels = categorize_by_panel(report)
        count = 0
        for panel_name, results in panels.items():
            lines.append(f"── {panel_name} ──")
            lines.append(f"{'Test':<30} {'Value':<12} {'Unit':<10} {'Range':<18} {'Flag'}")
            lines.append("─" * 85)
            for r in results:
                if count >= max_results:
                    lines.append(f"... ({len(report.results) - count} more results truncated)")
                    break
                flag_str = "" if r.flag == ResultFlag.NORMAL else f"⚠ {r.flag.value.upper()}"
                lines.append(
                    f"{r.test_name:<30} {r.value:<12} {r.unit:<10} {r.reference_range:<18} {flag_str}"
                )
                count += 1
            lines.append("")

    # Raw text fallback (for radiology / pathology / general reports)
    if report.raw_text and not report.results:
        lines.append("── Report Text ──")
        lines.append(report.raw_text[:5000])
        if len(report.raw_text) > 5000:
            lines.append("... (text truncated)")

    return "\n".join(lines)
