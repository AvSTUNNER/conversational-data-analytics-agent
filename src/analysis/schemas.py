"""Pydantic schemas and enums for deterministic analysis plans and results."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OperationType(str, Enum):
    COUNT = "count"
    AGGREGATION = "aggregation"
    GROUPBY_AGGREGATE = "groupby_aggregate"
    TREND = "trend"
    CORRELATION = "correlation"
    OUTLIERS = "outliers"
    FILTER_PREVIEW = "filter_preview"
    UNSUPPORTED = "unsupported"


class AggregationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD = "std"
    NUNIQUE = "nunique"


class FilterOperator(str, Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class FilterCondition(BaseModel):
    """Represents a single deterministic filter predicate."""
    model_config = ConfigDict(extra="ignore")

    column: str
    operator: FilterOperator
    value: Any

    @field_validator("operator", mode="before")
    @classmethod
    def normalize_operator(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_str = v.strip().lower()
            mapping = {
                "=": FilterOperator.EQUALS,
                "==": FilterOperator.EQUALS,
                "eq": FilterOperator.EQUALS,
                "equals": FilterOperator.EQUALS,
                "!=": FilterOperator.NOT_EQUALS,
                "neq": FilterOperator.NOT_EQUALS,
                "not_equals": FilterOperator.NOT_EQUALS,
                ">": FilterOperator.GREATER_THAN,
                "gt": FilterOperator.GREATER_THAN,
                ">=": FilterOperator.GREATER_EQUAL,
                "gte": FilterOperator.GREATER_EQUAL,
                "<": FilterOperator.LESS_THAN,
                "lt": FilterOperator.LESS_THAN,
                "<=": FilterOperator.LESS_EQUAL,
                "lte": FilterOperator.LESS_EQUAL,
                "in": FilterOperator.IN,
                "not_in": FilterOperator.NOT_IN,
                "contains": FilterOperator.CONTAINS,
                "like": FilterOperator.CONTAINS,
            }
            if v_str in mapping:
                return mapping[v_str]
        return v


class ChartRecommendation(str, Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    NONE = "none"


class TimeGranularity(str, Enum):
    DAY = "D"
    WEEK = "W"
    MONTH = "ME"
    QUARTER = "QE"
    YEAR = "YE"


class AnalysisPlan(BaseModel):
    """Rigid intermediate representation for analysis execution."""
    model_config = ConfigDict(extra="ignore")

    operation: OperationType
    target_column: Optional[str] = None
    target_columns: Optional[List[str]] = None
    group_by_columns: Optional[List[str]] = None
    aggregation: Optional[AggregationType] = None
    filters: Optional[List[FilterCondition]] = Field(default_factory=list)
    sort_ascending: Optional[bool] = None
    limit: Optional[int] = None
    date_column: Optional[str] = None
    time_granularity: Optional[TimeGranularity] = None
    chart: ChartRecommendation = ChartRecommendation.NONE
    unsupported_reason: Optional[str] = None
    explanation_focus: Optional[str] = None

    @field_validator("operation", mode="before")
    @classmethod
    def normalize_operation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower().replace("-", "_").replace(" ", "_")
            if v_clean in {"groupby", "group_by", "group_by_aggregate", "groupby_aggregate"}:
                return OperationType.GROUPBY_AGGREGATE
            if v_clean in {"aggregate", "aggregation", "metric"}:
                return OperationType.AGGREGATION
            if v_clean in {"outlier", "outliers", "anomaly", "anomalies"}:
                return OperationType.OUTLIERS
            if v_clean in {"filter", "preview", "filter_preview", "top_n", "topn"}:
                return OperationType.FILTER_PREVIEW
        return v

    @field_validator("aggregation", mode="before")
    @classmethod
    def normalize_aggregation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in {"avg", "average", "mean"}:
                return AggregationType.MEAN
            if v_clean in {"distinct", "distinct_count", "unique", "nunique"}:
                return AggregationType.NUNIQUE
            if v_clean in {"total", "sum"}:
                return AggregationType.SUM
        return v

    @field_validator("time_granularity", mode="before")
    @classmethod
    def normalize_granularity(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in {"d", "day", "daily"}:
                return TimeGranularity.DAY
            if v_clean in {"w", "week", "weekly"}:
                return TimeGranularity.WEEK
            if v_clean in {"m", "me", "month", "monthly"}:
                return TimeGranularity.MONTH
            if v_clean in {"q", "qe", "quarter", "quarterly"}:
                return TimeGranularity.QUARTER
            if v_clean in {"y", "ye", "year", "yearly", "annual", "annually"}:
                return TimeGranularity.YEAR
        return v

    @field_validator("chart", mode="before")
    @classmethod
    def normalize_chart(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower().replace("_chart", "").replace("chart", "").strip()
            for ch in ChartRecommendation:
                if ch.value == v_clean:
                    return ch
        return v

    @field_validator("group_by_columns", "target_columns", mode="before")
    @classmethod
    def normalize_list_columns(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = [v]
        return v

    @model_validator(mode="before")
    @classmethod
    def normalize_model_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support "group_by" alias
            if "group_by" in data and "group_by_columns" not in data:
                val = data.pop("group_by")
                data["group_by_columns"] = [val] if isinstance(val, str) else val
            # Support "target" alias
            if "target" in data and "target_column" not in data:
                data["target_column"] = data.pop("target")
            # Support "sort" alias ("descending" / "ascending")
            if "sort" in data and "sort_ascending" not in data:
                s_val = str(data.pop("sort")).strip().lower()
                if s_val in {"desc", "descending", "highest", "top"}:
                    data["sort_ascending"] = False
                elif s_val in {"asc", "ascending", "lowest", "bottom"}:
                    data["sort_ascending"] = True
            # Support "granularity" alias
            if "granularity" in data and "time_granularity" not in data:
                data["time_granularity"] = data.pop("granularity")
        return data


class ExecutionResult(BaseModel):
    """Structured container for deterministic execution output and evidence."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation: OperationType
    main_value: Optional[Union[float, int, str, Dict[str, Any]]] = None
    evidence_df: Optional[pd.DataFrame] = None
    evidence_summary: str = ""
    chart_type: ChartRecommendation = ChartRecommendation.NONE
    is_unsupported: bool = False
    unsupported_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
