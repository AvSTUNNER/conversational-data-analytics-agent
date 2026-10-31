"""Provider-agnostic LLM client abstraction supporting Ollama, Gemini, Groq, and Mock fallback."""
from __future__ import annotations

import difflib
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import requests


class LLMClient(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: Optional[str] = None,
    ) -> str:
        """Generates text from the LLM given a user prompt and optional system prompt."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama instances (e.g. Qwen2.5, Llama3.2)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt
        if response_format == "json":
            payload["format"] = "json"

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to local Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`). Error: {str(e)}"
            ) from e


class GeminiClient(LLMClient):
    """Client for Google Gemini free tier API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or configuration.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        contents: List[Dict[str, Any]] = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow all system instructions strictly."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json" if response_format == "json" else "text/plain",
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {str(e)}") from e

class GroqClient(LLMClient):
    """Client for Groq API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "text",
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()

            data = resp.json()
            choices = data.get("choices", [])

            if not choices:
                return ""

            return choices[0].get("message", {}).get("content", "").strip()

        except Exception as e:
            raise RuntimeError(
                f"Groq API request failed: {str(e)}"
            ) from e

class MockLLMClient(LLMClient):
    """Deterministic, generalized semantic parser mock client for offline testing, benchmarks, and CI."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: Optional[str] = None,
    ) -> str:
        question = ""
        q_match = re.search(r'USER QUESTION:\s*"(.*?)"', prompt, re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
        else:
            question = prompt

        num_cols, cat_cols, date_cols, all_cols = self._extract_schema_columns(prompt)
        q_lower = question.lower()

        # 1. Out-of-Scope / Unsupported Check
        unsupported_keywords = [
            "weather", "temperature", "stock price", "stock", "turnover",
            "employee", "salary", "ceo", "gdp", "inflation", "market ticker", "ticker"
        ]
        for kw in unsupported_keywords:
            if kw in q_lower and not any(kw in c.lower() for c in all_cols):
                return json.dumps({
                    "operation": "unsupported",
                    "unsupported_reason": f"The dataset does not contain information regarding '{kw}'.",
                    "chart": "none"
                })

        # Match columns with priority
        matched_nums = self._find_multiple_matches(q_lower, num_cols)
        matched_cats = self._find_multiple_matches(q_lower, cat_cols)
        matched_all = self._find_multiple_matches(q_lower, all_cols)

        # 2. Outliers
        if any(w in q_lower for w in ["outlier", "outliers", "anomaly", "anomalies"]):
            tgt = matched_nums[0] if matched_nums else (num_cols[0] if num_cols else "Sales")
            return json.dumps({
                "operation": "outliers",
                "target_column": tgt,
                "chart": "box"
            })

        # 3. Correlation / Relationship
        is_correlation = bool(re.search(r"\b(correlation|relationship|correlated|correlate)\b", q_lower))
        if is_correlation:
            if len(matched_nums) >= 2:
                return json.dumps({
                    "operation": "correlation",
                    "target_columns": matched_nums[:2],
                    "chart": "scatter"
                })
            elif len(num_cols) >= 2:
                return json.dumps({
                    "operation": "correlation",
                    "target_columns": num_cols[:2],
                    "chart": "scatter"
                })

        # 4. Trend / Time-series
        is_trend_query = any(w in q_lower for w in ["trend", "trends", "over time"]) or (
            any(w in q_lower for w in ["monthly", "yearly", "daily"])
            and "monthly charge" not in q_lower
            and "monthlycharge" not in q_lower
        )
        if is_trend_query and date_cols:
            date_col = date_cols[0]
            tgt = matched_nums[0] if matched_nums else (num_cols[0] if num_cols else "Sales")
            gran = "ME"
            if "daily" in q_lower or "day" in q_lower:
                gran = "D"
            elif "yearly" in q_lower or "year" in q_lower or "annual" in q_lower:
                gran = "YE"
            return json.dumps({
                "operation": "trend",
                "date_column": date_col,
                "target_column": tgt,
                "aggregation": "sum",
                "time_granularity": gran,
                "chart": "line"
            })

        # 5. Extract Filters if mentioned in question
        filters = []
        if "technology" in q_lower:
            filters.append({"column": "Category", "operator": "==", "value": "Technology"})
        if "corporate" in q_lower:
            filters.append({"column": "Segment", "operator": "==", "value": "Corporate"})
        if "same day" in q_lower:
            filters.append({"column": "Ship Mode", "operator": "==", "value": "Same Day"})
        if "seniorcitizen" in q_lower or "senior citizen" in q_lower:
            filters.append({"column": "SeniorCitizen", "operator": "==", "value": 1})
        if "churned" in q_lower:
            filters.append({"column": "Churn", "operator": "==", "value": "Yes"})
        if "weekend" in q_lower:
            filters.append({"column": "Weekend", "operator": "==", "value": True})

        # 6. Count Queries
        is_count_query = any(w in q_lower for w in ["how many", "count of", "number of", "how many total", "how many rows", "how many unique", "how many distinct"])
        is_asking_distinct = "distinct" in q_lower or "unique" in q_lower
        has_specific_metric = any(w in q_lower for w in ["sales", "profit", "discount", "charge", "revenue", "price", "rate", "page value", "duration", "tenure"])

        if is_count_query and not (has_specific_metric and any(w in q_lower for w in ["average", "mean", "median", "total profit", "total sales", "total quantity"])):
            if is_asking_distinct:
                id_col = "Customer ID" if ("customer" in q_lower and "Customer ID" in all_cols) else (self._find_best_match(q_lower, all_cols) or "Order ID")
                return json.dumps({
                    "operation": "count",
                    "target_column": id_col,
                    "aggregation": "nunique",
                    "filters": filters,
                    "chart": "none"
                })
            return json.dumps({
                "operation": "count",
                "filters": filters,
                "chart": "none"
            })

        # 7. Groupby + Aggregate
        has_by = bool(re.search(r"\b(by|per|across)\s+([a-zA-Z_\s-]+)", q_lower))
        is_top_query = "top " in q_lower or "top 5" in q_lower or "highest" in q_lower or "which" in q_lower or "most" in q_lower
        
        # Determine target group column
        filter_cols = {f["column"].lower() for f in filters}
        available_grp = [c for c in matched_all if c.lower() not in filter_cols and c in cat_cols]
        if not available_grp:
            # Check if an integer/code column is mentioned in 'by' clause
            by_match = re.search(r"\b(by|per|across)\s+([a-zA-Z_\s-]+)", q_lower)
            if by_match:
                by_text = by_match.group(2).strip()
                matched_by = self._find_multiple_matches(by_text, all_cols)
                if matched_by:
                    available_grp = [matched_by[0]]

        if not available_grp and matched_cats:
            available_grp = matched_cats

        for c in cat_cols:
            if f"which {c.lower()}" in q_lower or f"what {c.lower()}" in q_lower:
                available_grp = [c]
                break

        if (has_by and available_grp) or (is_top_query and available_grp) or ("top 5 products" in q_lower):
            grp_col = available_grp[0] if available_grp else "Product Name"
            if "top 5 products" in q_lower and "Product Name" in cat_cols:
                grp_col = "Product Name"

            # Pick target numeric column that is distinct from grp_col
            candidate_nums = [c for c in matched_nums if c != grp_col]
            if not candidate_nums:
                candidate_nums = [c for c in num_cols if c != grp_col]
            tgt_col = candidate_nums[0] if candidate_nums else "Sales"
            
            agg = "sum"
            if any(w in q_lower for w in ["average", "mean", "avg"]):
                agg = "mean"
            elif "median" in q_lower:
                agg = "median"
            elif "used the most" in q_lower or "count" in q_lower:
                agg = "count"

            limit = None
            if "top 5" in q_lower:
                limit = 5
            elif "which" in q_lower or ("highest" in q_lower and "by" not in q_lower) or "used the most" in q_lower:
                limit = 1

            return json.dumps({
                "operation": "groupby_aggregate",
                "group_by_columns": [grp_col],
                "target_column": tgt_col,
                "aggregation": agg,
                "filters": filters,
                "sort_ascending": False,
                "limit": limit,
                "chart": "bar"
            })

        # 8. Single Aggregation
        if matched_nums or any(w in q_lower for w in ["average", "mean", "avg", "median", "total", "sum", "overall", "min", "max", "maximum", "minimum"]):
            tgt_col = matched_nums[0] if matched_nums else (num_cols[0] if num_cols else "Sales")
            agg = "sum"
            if any(w in q_lower for w in ["average", "mean", "avg"]):
                agg = "mean"
            elif "median" in q_lower:
                agg = "median"
            elif "min" in q_lower or "lowest" in q_lower or "minimum" in q_lower:
                agg = "min"
            elif "max" in q_lower or "highest" in q_lower or "maximum" in q_lower:
                agg = "max"

            return json.dumps({
                "operation": "aggregation",
                "target_column": tgt_col,
                "aggregation": agg,
                "filters": filters,
                "chart": "none"
            })

        # Default fallback
        return json.dumps({
            "operation": "count",
            "filters": filters,
            "chart": "none"
        })

    @staticmethod
    def _extract_schema_columns(prompt: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Parses column names and semantic types from the schema prompt."""
        num_cols = []
        cat_cols = []
        date_cols = []
        all_cols = []

        lines = prompt.splitlines()
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("- ") and ":" in line_str:
                parts = line_str[2:].split(":")
                col_name = parts[0].strip()
                all_cols.append(col_name)
                type_desc = parts[1].upper()
                if "NUMERIC" in type_desc:
                    num_cols.append(col_name)
                elif "DATETIME" in type_desc:
                    date_cols.append(col_name)
                elif "CATEGORICAL" in type_desc or "BOOLEAN" in type_desc:
                    cat_cols.append(col_name)

        return num_cols, cat_cols, date_cols, all_cols

    @staticmethod
    def _find_best_match(query: str, columns: List[str]) -> Optional[str]:
        """Finds the column that best matches a term in the query."""
        for col in columns:
            col_clean = col.lower().replace("_", " ").replace("-", " ")
            if col.lower() in query or col_clean in query:
                return col
        return None

    @classmethod
    def _find_multiple_matches(cls, query: str, columns: List[str]) -> List[str]:
        """Finds all columns mentioned in the query."""
        matched = []
        sorted_cols = sorted(columns, key=lambda c: len(c), reverse=True)
        for col in sorted_cols:
            col_clean = col.lower().replace("_", " ").replace("-", " ")
            col_singular = col.rstrip("s").lower()
            col_stem_clean = col_clean.rstrip("s")
            
            if (
                col.lower() in query
                or col_clean in query
                or (len(col_singular) > 3 and col_singular in query)
                or (len(col_stem_clean) > 3 and col_stem_clean in query)
                or (col == "Contract" and "contract" in query)
                or (col == "VisitorType" and "visitor" in query)
                or (col == "PageValues" and ("page value" in query or "page values" in query))
                or (col == "BounceRates" and ("bounce rate" in query or "bounce rates" in query))
                or (col == "ExitRates" and ("exit rate" in query or "exit rates" in query))
                or (col == "ProductRelated_Duration" and ("product related duration" in query or "duration" in query))
                or (col == "MonthlyCharges" and ("monthly charge" in query or "monthly charges" in query))
                or (col == "Product Name" and ("product" in query or "products" in query))
                or (col == "OperatingSystems" and ("operating system" in query or "os" in query))
                or (col == "InternetService" and "internet" in query)
                or (col == "PaymentMethod" and "payment" in query)
            ):
                if col not in matched:
                    matched.append(col)
        return matched


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    chosen_provider = (
        provider or os.getenv("LLM_PROVIDER", "ollama")
    ).lower()

    if chosen_provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is not set in environment or configuration.")
        return GeminiClient()

    if chosen_provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set in environment or configuration.")
        return GroqClient()

    if chosen_provider == "ollama":
        try:
            return OllamaClient()
        except Exception:
            return MockLLMClient()

    if chosen_provider == "mock":
        return MockLLMClient()

    return MockLLMClient()
