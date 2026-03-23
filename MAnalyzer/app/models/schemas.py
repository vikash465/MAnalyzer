from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ReportType(str, Enum):
    LAB = "lab"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    GENERAL = "general"


class ResultFlag(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    CRITICAL_HIGH = "critical_high"
    CRITICAL_LOW = "critical_low"
    ABNORMAL = "abnormal"


class PatientInfo(BaseModel):
    name: str = "Unknown"
    age: int | None = None
    gender: str | None = None
    patient_id: str | None = None


class LabResult(BaseModel):
    test_name: str
    value: str
    unit: str = ""
    reference_range: str = ""
    flag: ResultFlag = ResultFlag.NORMAL


class MedicalReport(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    report_type: ReportType = ReportType.GENERAL
    report_date: str = ""
    results: list[LabResult] = Field(default_factory=list)
    raw_text: str = ""


# --- API request / response models ---

class UploadReportResponse(BaseModel):
    session_id: str
    filename: str
    report_type: str
    patient: PatientInfo
    result_count: int
    abnormal_count: int
    preview: list[LabResult]


class AnalysisRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User question or message",
    )
    provider: str | None = Field(
        default=None,
        pattern=r"^(openai|claude)$",
        description="LLM provider override (openai, claude)",
    )


class AnalysisResponse(BaseModel):
    reply: str
    provider: str
    disclaimer: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=None))


class ReportSummary(BaseModel):
    session_id: str
    filename: str
    report_type: str
    patient: PatientInfo
    total_tests: int
    normal_count: int
    abnormal_count: int
    all_results: list[LabResult]
    disclaimer: str = ""


class ErrorResponse(BaseModel):
    detail: str
