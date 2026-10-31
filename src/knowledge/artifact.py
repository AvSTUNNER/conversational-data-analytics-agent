"""Dynamic knowledge layer for dataset domain knowledge, data dictionaries, and contextual artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class KnowledgeSourceType(str, Enum):
    """Origin of the dataset knowledge artifact."""
    SUPPLIED = "supplied"
    GENERATED = "generated"


class ColumnKnowledge(BaseModel):
    """Detailed observed metadata and optional inferred semantic role for a column."""
    name: str
    data_type: str
    raw_dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list)
    numeric_range: Optional[str] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    date_span: Optional[str] = None
    inferred_role: Optional[str] = None  # Explicitly labeled as inferred


class ObservedMetadata(BaseModel):
    """Strictly observed dataset facts extracted directly from DatasetProfile and DataFrame."""
    dataset_name: str
    row_count: int
    column_count: int
    duplicate_rows: int
    total_null_values: int
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    boolean_columns: List[str] = Field(default_factory=list)
    columns: Dict[str, ColumnKnowledge] = Field(default_factory=dict)


class InferredMetadata(BaseModel):
    """Heuristic or inferred semantic characteristics, strictly tagged as INFERRED."""
    suggested_domain: Optional[str] = None
    inferred_entity_identifiers: List[str] = Field(default_factory=list)
    inferred_metric_columns: List[str] = Field(default_factory=list)
    inferred_categorical_dimensions: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class KnowledgeArtifact(BaseModel):
    """Authoritative representation of dataset-specific domain knowledge and data dictionary."""
    dataset_name: str
    source_type: KnowledgeSourceType
    observed: ObservedMetadata
    inferred: InferredMetadata
    supplied_content: Optional[str] = None
    supplied_filename: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_full_text(self) -> str:
        """Renders a structured, clean markdown representation of the knowledge artifact."""
        lines: List[str] = []
        lines.append(f"# Knowledge Artifact: {self.dataset_name}")
        lines.append(f"**Source Type:** {self.source_type.value.upper()}")
        lines.append(f"**Generated / Updated:** {self.created_at}")
        lines.append("")

        if self.source_type == KnowledgeSourceType.SUPPLIED and self.supplied_content:
            lines.append("## User-Supplied Knowledge Document")
            if self.supplied_filename:
                lines.append(f"*(Uploaded from: `{self.supplied_filename}`)*")
            lines.append("")
            lines.append(self.supplied_content.strip())
            lines.append("")
            lines.append("---")
            lines.append("## Underlying Observed Dataset Facts (Authoritative Ground Truth)")
        else:
            lines.append("## Observed Dataset Facts (Authoritative Ground Truth)")

        lines.append(f"- **Total Rows:** {self.observed.row_count:,}")
        lines.append(f"- **Total Columns:** {self.observed.column_count}")
        lines.append(f"- **Duplicate Rows:** {self.observed.duplicate_rows:,}")
        lines.append(f"- **Total Null Values:** {self.observed.total_null_values:,}")
        lines.append("")
        lines.append("### Column Data Dictionary")
        lines.append("| Column Name | Type | Raw Dtype | Nulls (%) | Unique Values | Details / Range | Inferred Role |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for col_name, col in self.observed.columns.items():
            details = []
            if col.numeric_range:
                details.append(f"Range: {col.numeric_range}")
            if col.mean is not None:
                details.append(f"Mean: {col.mean:.2f}")
            if col.date_span:
                details.append(f"Span: {col.date_span}")
            if not details and col.sample_values:
                samples = ", ".join(repr(v) for v in col.sample_values[:3])
                details.append(f"Samples: [{samples}]")
            details_str = "; ".join(details) if details else "None"

            role_str = f"[INFERRED] {col.inferred_role}" if col.inferred_role else "None"
            lines.append(
                f"| `{col_name}` | {col.data_type.upper()} | {col.raw_dtype} | "
                f"{col.null_count} ({col.null_percentage:.1f}%) | {col.unique_count:,} | "
                f"{details_str} | {role_str} |"
            )

        lines.append("")
        lines.append("## Inferred Semantic Notes (Non-Authoritative)")
        if self.inferred.suggested_domain:
            lines.append(f"- **Suggested Domain:** [INFERRED] {self.inferred.suggested_domain}")
        if self.inferred.inferred_entity_identifiers:
            lines.append(f"- **Inferred Potential IDs:** [INFERRED] {', '.join(self.inferred.inferred_entity_identifiers)}")
        if self.inferred.inferred_metric_columns:
            lines.append(f"- **Inferred Value Metrics:** [INFERRED] {', '.join(self.inferred.inferred_metric_columns)}")
        if self.inferred.inferred_categorical_dimensions:
            lines.append(f"- **Inferred Grouping Dimensions:** [INFERRED] {', '.join(self.inferred.inferred_categorical_dimensions)}")
        for note in self.inferred.notes:
            lines.append(f"- [INFERRED] {note}")

        return "\n".join(lines)
