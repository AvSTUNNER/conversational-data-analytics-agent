"""Deterministic chunking for dataset knowledge artifacts and user-supplied documents."""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.knowledge.artifact import KnowledgeArtifact, KnowledgeSourceType


class DocumentChunk(BaseModel):
    """A discrete, indexed segment of knowledge with strict dataset provenance."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_name: str
    source_type: KnowledgeSourceType
    section_title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def full_representation(self) -> str:
        """Returns text representation optimized for vector embedding and retrieval."""
        return f"[{self.dataset_name}] ({self.section_title})\n{self.content}"


class DocumentChunker:
    """Chunks structured knowledge artifacts into semantic segments preserving headings, dictionary entries, and provenance."""

    @classmethod
    def chunk_artifact(cls, artifact: KnowledgeArtifact) -> List[DocumentChunk]:
        """Splits a KnowledgeArtifact into structured, traceable chunks."""
        chunks: List[DocumentChunk] = []

        # 1. Dataset Overview Chunk (Observed Facts)
        overview_content = (
            f"Dataset: {artifact.dataset_name}\n"
            f"Total Rows: {artifact.observed.row_count:,}\n"
            f"Total Columns: {artifact.observed.column_count}\n"
            f"Duplicate Rows: {artifact.observed.duplicate_rows:,}\n"
            f"Total Null Values: {artifact.observed.total_null_values:,}\n"
            f"Numeric Columns: {', '.join(artifact.observed.numeric_columns)}\n"
            f"Categorical Columns: {', '.join(artifact.observed.categorical_columns)}\n"
            f"Datetime Columns: {', '.join(artifact.observed.datetime_columns)}\n"
            f"Boolean Columns: {', '.join(artifact.observed.boolean_columns)}"
        )
        chunks.append(
            DocumentChunk(
                dataset_name=artifact.dataset_name,
                source_type=artifact.source_type,
                section_title="Dataset Overview & Schema Facts",
                content=overview_content,
                metadata={
                    "type": "overview",
                    "row_count": artifact.observed.row_count,
                    "column_count": artifact.observed.column_count,
                },
            )
        )

        # 2. User-Supplied Knowledge Document Chunks (if present)
        if artifact.source_type == KnowledgeSourceType.SUPPLIED and artifact.supplied_content:
            supplied_chunks = cls._chunk_supplied_text(
                dataset_name=artifact.dataset_name,
                content=artifact.supplied_content,
                filename=artifact.supplied_filename or "user_doc.txt",
            )
            chunks.extend(supplied_chunks)

        # 3. Column Data Dictionary Chunks (Granular per column for precise semantic matching)
        for col_name, col_meta in artifact.observed.columns.items():
            col_lines = [
                f"Column: {col_name}",
                f"Data Type: {col_meta.data_type.upper()} ({col_meta.raw_dtype})",
                f"Null Count: {col_meta.null_count} ({col_meta.null_percentage:.1f}%)",
                f"Unique Values: {col_meta.unique_count:,}",
            ]
            if col_meta.numeric_range:
                col_lines.append(f"Numeric Range: {col_meta.numeric_range}")
            if col_meta.mean is not None:
                col_lines.append(f"Mean: {col_meta.mean:.2f}")
            if col_meta.median is not None:
                col_lines.append(f"Median: {col_meta.median:g}")
            if col_meta.date_span:
                col_lines.append(f"Date Span: {col_meta.date_span}")
            if col_meta.sample_values:
                col_lines.append(f"Sample Values: {', '.join(repr(s) for s in col_meta.sample_values[:5])}")
            if col_meta.inferred_role:
                col_lines.append(f"Inferred Semantic Role: [INFERRED] {col_meta.inferred_role}")

            chunks.append(
                DocumentChunk(
                    dataset_name=artifact.dataset_name,
                    source_type=artifact.source_type,
                    section_title=f"Column Dictionary: {col_name}",
                    content="\n".join(col_lines),
                    metadata={
                        "type": "column_dictionary",
                        "column_name": col_name,
                        "data_type": col_meta.data_type,
                        "inferred_role": col_meta.inferred_role,
                    },
                )
            )

        # 4. Inferred Semantic Notes Chunk
        inferred_lines = []
        if artifact.inferred.suggested_domain:
            inferred_lines.append(f"Suggested Business Domain: [INFERRED] {artifact.inferred.suggested_domain}")
        if artifact.inferred.inferred_entity_identifiers:
            inferred_lines.append(f"Potential Unique Identifiers: [INFERRED] {', '.join(artifact.inferred.inferred_entity_identifiers)}")
        if artifact.inferred.inferred_metric_columns:
            inferred_lines.append(f"Key Quantitative Metrics: [INFERRED] {', '.join(artifact.inferred.inferred_metric_columns)}")
        if artifact.inferred.inferred_categorical_dimensions:
            inferred_lines.append(f"Categorical Dimensions: [INFERRED] {', '.join(artifact.inferred.inferred_categorical_dimensions)}")
        for note in artifact.inferred.notes:
            inferred_lines.append(f"Note: [INFERRED] {note}")

        if inferred_lines:
            chunks.append(
                DocumentChunk(
                    dataset_name=artifact.dataset_name,
                    source_type=artifact.source_type,
                    section_title="Inferred Semantic Notes",
                    content="\n".join(inferred_lines),
                    metadata={"type": "inferred_semantics", "is_inferred": True},
                )
            )

        return chunks

    @classmethod
    def _chunk_supplied_text(
        cls,
        dataset_name: str,
        content: str,
        filename: str,
    ) -> List[DocumentChunk]:
        """Splits markdown/text supplied document by headers or logical paragraphs."""
        chunks: List[DocumentChunk] = []
        # Split on markdown headers (#, ##, ###) or double newlines
        header_pattern = r"(?m)^(#{1,4}\s+.*$)"
        sections = re.split(header_pattern, content)

        if len(sections) == 1:
            # No markdown headers, split by double newlines into paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, p in enumerate(paragraphs):
                chunks.append(
                    DocumentChunk(
                        dataset_name=dataset_name,
                        source_type=KnowledgeSourceType.SUPPLIED,
                        section_title=f"User Document: {filename} (Section {i+1})",
                        content=p,
                        metadata={"type": "user_supplied", "filename": filename, "paragraph_index": i},
                    )
                )
            return chunks

        # Structured markdown with headers
        current_title = f"User Document: {filename}"
        for part in sections:
            part = part.strip()
            if not part:
                continue
            if part.startswith("#"):
                current_title = f"{filename} > {part.lstrip('#').strip()}"
            else:
                chunks.append(
                    DocumentChunk(
                        dataset_name=dataset_name,
                        source_type=KnowledgeSourceType.SUPPLIED,
                        section_title=current_title,
                        content=part,
                        metadata={"type": "user_supplied", "filename": filename},
                    )
                )

        return chunks
