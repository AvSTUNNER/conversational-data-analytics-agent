"""Knowledge manager responsible for auto-generating, uploading, and strictly isolating dataset knowledge artifacts."""
from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from src.data.profiler import DatasetProfile
from src.knowledge.artifact import (
    ColumnKnowledge,
    InferredMetadata,
    KnowledgeArtifact,
    KnowledgeSourceType,
    ObservedMetadata,
)


class KnowledgeManager:
    """Manages lifecycle, auto-generation, user uploads, and dataset isolation for knowledge artifacts."""

    def __init__(self) -> None:
        self._current_artifact: Optional[KnowledgeArtifact] = None
        self._current_dataset_id: Optional[str] = None
        # In-memory storage keyed by unique dataset identifier to prevent leakage
        self._dataset_artifacts: Dict[str, KnowledgeArtifact] = {}

    @property
    def current_artifact(self) -> Optional[KnowledgeArtifact]:
        """Returns the active knowledge artifact for the currently active dataset."""
        return self._current_artifact

    def set_active_dataset(
        self,
        dataset_name: str,
        profile: DatasetProfile,
        df: Optional[pd.DataFrame] = None,
    ) -> KnowledgeArtifact:
        """Sets the active dataset, isolating and loading or generating its knowledge artifact.

        If a knowledge artifact already exists for this dataset_name (e.g. user previously supplied one),
        it is retrieved. Otherwise, an artifact is deterministically generated from facts.
        """
        self._current_dataset_id = dataset_name

        if dataset_name in self._dataset_artifacts:
            # Re-activate stored artifact for this dataset
            self._current_artifact = self._dataset_artifacts[dataset_name]
            # Refresh underlying observed numbers if profile changed
            self._refresh_observed_metadata(self._current_artifact, dataset_name, profile)
        else:
            # Generate fresh factual artifact
            artifact = self.generate_artifact(dataset_name, profile, df)
            self._dataset_artifacts[dataset_name] = artifact
            self._current_artifact = artifact

        return self._current_artifact

    def clear_active_dataset(self) -> None:
        """Explicitly clears active dataset knowledge to prevent cross-dataset leakage."""
        self._current_artifact = None
        self._current_dataset_id = None

    def upload_supplied_knowledge(
        self,
        dataset_name: str,
        content: str,
        filename: str,
        profile: DatasetProfile,
        df: Optional[pd.DataFrame] = None,
    ) -> KnowledgeArtifact:
        """Sets user-supplied knowledge as authoritative for the dataset, preserving observed facts."""
        if not content or not content.strip():
            raise ValueError("Supplied knowledge content cannot be empty.")

        # Ensure active dataset matches
        if self._current_dataset_id != dataset_name or self._current_artifact is None:
            self.set_active_dataset(dataset_name, profile, df)

        base_artifact = self.generate_artifact(dataset_name, profile, df)
        supplied_artifact = KnowledgeArtifact(
            dataset_name=dataset_name,
            source_type=KnowledgeSourceType.SUPPLIED,
            observed=base_artifact.observed,
            inferred=base_artifact.inferred,
            supplied_content=content.strip(),
            supplied_filename=filename,
        )

        self._dataset_artifacts[dataset_name] = supplied_artifact
        self._current_artifact = supplied_artifact
        return supplied_artifact

    def reset_to_generated(
        self,
        dataset_name: str,
        profile: DatasetProfile,
        df: Optional[pd.DataFrame] = None,
    ) -> KnowledgeArtifact:
        """Reverts user-supplied knowledge back to purely generated factual knowledge."""
        artifact = self.generate_artifact(dataset_name, profile, df)
        self._dataset_artifacts[dataset_name] = artifact
        self._current_artifact = artifact
        return artifact

    def generate_artifact(
        self,
        dataset_name: str,
        profile: DatasetProfile,
        df: Optional[pd.DataFrame] = None,
    ) -> KnowledgeArtifact:
        """Generates a knowledge artifact deterministically from observed dataset facts.

        Observed facts are purely derived from DatasetProfile and DataFrame.
        Any heuristic semantic classification is strictly flagged as INFERRED.
        """
        # 1. Build Observed Column Metadata
        columns_meta: Dict[str, ColumnKnowledge] = {}
        for col_name, col_prof in profile.columns.items():
            numeric_range = None
            mean_val = None
            median_val = None
            date_span = None

            if col_prof.numeric_stats:
                st = col_prof.numeric_stats
                numeric_range = f"{st.min:g} to {st.max:g}"
                mean_val = st.mean
                median_val = st.median
            elif col_prof.date_range:
                date_span = f"{col_prof.date_range.min_date} to {col_prof.date_range.max_date}"

            # Simple heuristic semantic role - strictly marked as INFERRED
            inferred_role = self._infer_column_role(col_prof.name, col_prof.data_type, col_prof.unique_count, profile.row_count)

            columns_meta[col_name] = ColumnKnowledge(
                name=col_prof.name,
                data_type=col_prof.data_type,
                raw_dtype=col_prof.raw_dtype,
                null_count=col_prof.null_count,
                null_percentage=col_prof.null_percentage,
                unique_count=col_prof.unique_count,
                sample_values=col_prof.sample_values,
                numeric_range=numeric_range,
                mean=mean_val,
                median=median_val,
                date_span=date_span,
                inferred_role=inferred_role,
            )

        observed = ObservedMetadata(
            dataset_name=dataset_name,
            row_count=profile.row_count,
            column_count=profile.column_count,
            duplicate_rows=profile.duplicate_rows,
            total_null_values=profile.total_null_values,
            numeric_columns=profile.numeric_columns,
            categorical_columns=profile.categorical_columns,
            datetime_columns=profile.datetime_columns,
            boolean_columns=profile.boolean_columns,
            columns=columns_meta,
        )

        # 2. Build Inferred Metadata
        inferred = self._build_inferred_metadata(dataset_name, profile)

        return KnowledgeArtifact(
            dataset_name=dataset_name,
            source_type=KnowledgeSourceType.GENERATED,
            observed=observed,
            inferred=inferred,
        )

    def _refresh_observed_metadata(
        self,
        artifact: KnowledgeArtifact,
        dataset_name: str,
        profile: DatasetProfile,
    ) -> None:
        """Updates observed numbers in an existing artifact when the underlying data is refreshed."""
        fresh = self.generate_artifact(dataset_name, profile)
        artifact.observed = fresh.observed

    def _infer_column_role(
        self,
        name: str,
        data_type: str,
        unique_count: int,
        row_count: int,
    ) -> Optional[str]:
        """Assigns an inferred semantic role to a column based on name and cardinality."""
        lower = name.lower()
        if "id" in lower or "code" in lower or (unique_count == row_count and row_count > 0):
            return "Potential Unique Identifier / Entity Key"
        if data_type == "numeric":
            if any(k in lower for k in ["sales", "profit", "revenue", "price", "charge", "amount", "cost", "budget"]):
                return "Key Financial / Business Measure"
            if any(k in lower for k in ["quantity", "count", "num", "duration", "tenure", "units"]):
                return "Quantitative / Volume Metric"
            return "Numeric Measure"
        if data_type == "datetime":
            return "Temporal Event / Timestamp"
        if data_type == "categorical":
            if unique_count <= 10:
                return "Low-Cardinality Category / Segmentation Dimension"
            return "Categorical Attribute / Grouping Dimension"
        if data_type == "boolean":
            return "Binary Flag / Indicator"
        return None

    def _build_inferred_metadata(
        self,
        dataset_name: str,
        profile: DatasetProfile,
    ) -> InferredMetadata:
        """Constructs high-level inferred metadata notes."""
        ids: List[str] = []
        metrics: List[str] = []
        dims: List[str] = []

        for name, prof in profile.columns.items():
            lower = name.lower()
            if "id" in lower or prof.unique_count == profile.row_count:
                ids.append(name)
            elif prof.data_type == "numeric":
                metrics.append(name)
            elif prof.data_type in ["categorical", "boolean"]:
                dims.append(name)

        # Domain inference heuristic
        lower_ds = dataset_name.lower()
        domain = "General Tabular Data"
        if "retail" in lower_ds or "superstore" in lower_ds or "shopper" in lower_ds:
            domain = "E-Commerce & Retail Sales"
        elif "churn" in lower_ds or "telco" in lower_ds:
            domain = "Telecommunications Customer Operations"
        elif "business" in lower_ds:
            domain = "Enterprise Operations & Financials"

        notes = [
            f"Dataset contains {profile.column_count} fields across {profile.row_count:,} records.",
            "All numerical aggregates should be computed directly by Pandas; this document serves as descriptive context.",
        ]
        if profile.total_null_values > 0:
            notes.append(f"Observed {profile.total_null_values:,} null data points requiring standard data handling.")

        return InferredMetadata(
            suggested_domain=domain,
            inferred_entity_identifiers=ids[:5],
            inferred_metric_columns=metrics[:5],
            inferred_categorical_dimensions=dims[:5],
            notes=notes,
        )
