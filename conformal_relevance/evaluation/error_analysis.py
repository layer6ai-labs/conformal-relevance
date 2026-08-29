"""Error analysis and categorization for scoring failures."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence
import re


class ErrorCategory(str, Enum):
    """High-level error categories."""

    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    CONTEXT_LENGTH = "context_length"
    CONTENT_FILTER = "content_filter"
    AUTHENTICATION = "authentication"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"  # Transient, likely recoverable with retry
    MEDIUM = "medium"  # May require configuration change
    HIGH = "high"  # Requires investigation
    CRITICAL = "critical"  # Blocks all processing


@dataclass
class ErrorInfo:
    """Detailed error information for a single failure."""

    sample_id: int
    run_id: int | None
    category: ErrorCategory
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime | None = None
    retry_count: int = 0
    raw_response: str | None = None
    intent: str | None = None
    input_length: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "category": self.category.value,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "retry_count": self.retry_count,
            "raw_response": self.raw_response,
            "intent": self.intent,
            "input_length": self.input_length,
        }


@dataclass
class ErrorPattern:
    """A recurring error pattern with frequency info."""

    pattern: str
    category: ErrorCategory
    count: int
    sample_ids: list[int]
    example_message: str
    suggested_action: str


@dataclass
class ErrorAnalysisResult:
    """Aggregated error analysis results."""

    total_errors: int
    total_samples: int
    error_rate: float
    by_category: dict[ErrorCategory, int]
    by_severity: dict[ErrorSeverity, int]
    by_type: dict[str, int]
    patterns: list[ErrorPattern]
    errors: list[ErrorInfo]
    timeline: list[tuple[datetime, int]] | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return 1.0 - self.error_rate


# Error pattern matching rules
ERROR_PATTERNS: list[tuple[str, ErrorCategory, str, ErrorSeverity, str]] = [
    # Rate limiting
    (r"rate.?limit|too.?many.?requests|429|quota.?exceeded",
     ErrorCategory.RATE_LIMIT, "rate_limit_exceeded", ErrorSeverity.LOW,
     "Reduce concurrency or add delay between requests"),

    # Timeout errors
    (r"timeout|timed?.?out|deadline.?exceeded|request.?took.?too.?long",
     ErrorCategory.TIMEOUT, "request_timeout", ErrorSeverity.LOW,
     "Increase timeout or reduce input length"),

    # Context length / token limit
    (r"context.?length|token.?limit|maximum.?context|too.?long|max.?tokens",
     ErrorCategory.CONTEXT_LENGTH, "context_length_exceeded", ErrorSeverity.MEDIUM,
     "Reduce input length or use chunking"),

    # Content filter / safety
    (r"content.?filter|safety|blocked|harmful|inappropriate|refused",
     ErrorCategory.CONTENT_FILTER, "content_filtered", ErrorSeverity.MEDIUM,
     "Review input content for policy violations"),

    # Authentication errors
    (r"authentication|unauthorized|invalid.?api.?key|401|forbidden|403",
     ErrorCategory.AUTHENTICATION, "auth_failed", ErrorSeverity.CRITICAL,
     "Check API key configuration"),

    # Network errors
    (r"connection.?error|network|dns|unreachable|connection.?refused|502|503|504",
     ErrorCategory.NETWORK_ERROR, "network_error", ErrorSeverity.LOW,
     "Check network connectivity, retry later"),

    # JSON parsing errors
    (r"json|parse|decode|invalid.?json|expecting.?value|unterminated",
     ErrorCategory.PARSING_ERROR, "json_parse_error", ErrorSeverity.MEDIUM,
     "LLM returned malformed JSON, may need prompt adjustment"),

    # Missing keys / validation
    (r"missing.?key|key.?error|required.?field|validation|missing.?score",
     ErrorCategory.VALIDATION_ERROR, "missing_required_field", ErrorSeverity.MEDIUM,
     "LLM response missing expected fields"),

    # Score format errors
    (r"score.?format|invalid.?score|score.?out.?of.?range|not.?a.?number",
     ErrorCategory.VALIDATION_ERROR, "invalid_score_format", ErrorSeverity.MEDIUM,
     "LLM returned score in unexpected format"),

    # Empty response
    (r"empty.?response|no.?content|blank|null.?response",
     ErrorCategory.PARSING_ERROR, "empty_response", ErrorSeverity.MEDIUM,
     "LLM returned empty response, may need retry"),

    # API-specific errors
    (r"api.?error|service.?unavailable|internal.?error|500",
     ErrorCategory.API_ERROR, "api_internal_error", ErrorSeverity.LOW,
     "Transient API error, retry should resolve"),

    # Model errors
    (r"model.?not.?found|invalid.?model|model.?overloaded",
     ErrorCategory.API_ERROR, "model_error", ErrorSeverity.HIGH,
     "Check model name or try different model"),
]


def categorize_error(
    error_message: str,
    sample_id: int = 0,
    run_id: int | None = None,
    timestamp: datetime | None = None,
    retry_count: int = 0,
    raw_response: str | None = None,
    intent: str | None = None,
    input_length: int | None = None,
) -> ErrorInfo:
    """Categorize an error based on its message.

    Args:
        error_message: The error message to categorize.
        sample_id: ID of the sample that failed.
        run_id: ID of the run (optional).
        timestamp: When the error occurred.
        retry_count: Number of retries attempted.
        raw_response: Raw LLM response if available.
        intent: Intent/query for the sample.
        input_length: Length of input text.

    Returns:
        ErrorInfo with categorization details.
    """
    message_lower = error_message.lower()

    for pattern, category, error_type, severity, _ in ERROR_PATTERNS:
        if re.search(pattern, message_lower, re.IGNORECASE):
            return ErrorInfo(
                sample_id=sample_id,
                run_id=run_id,
                category=category,
                error_type=error_type,
                message=error_message,
                severity=severity,
                timestamp=timestamp,
                retry_count=retry_count,
                raw_response=raw_response,
                intent=intent,
                input_length=input_length,
            )

    # Unknown error
    return ErrorInfo(
        sample_id=sample_id,
        run_id=run_id,
        category=ErrorCategory.UNKNOWN,
        error_type="unknown_error",
        message=error_message,
        severity=ErrorSeverity.HIGH,
        timestamp=timestamp,
        retry_count=retry_count,
        raw_response=raw_response,
        intent=intent,
        input_length=input_length,
    )


def get_suggested_action(category: ErrorCategory) -> str:
    """Get suggested action for an error category.

    Args:
        category: The error category.

    Returns:
        Suggested remediation action.
    """
    actions = {
        ErrorCategory.API_ERROR: "Retry the request or check API status",
        ErrorCategory.RATE_LIMIT: "Reduce concurrency or add delay between requests",
        ErrorCategory.TIMEOUT: "Increase timeout or reduce input length",
        ErrorCategory.PARSING_ERROR: "Check LLM response format, adjust prompt",
        ErrorCategory.VALIDATION_ERROR: "Review prompt template for missing instructions",
        ErrorCategory.CONTEXT_LENGTH: "Reduce input length or use chunking",
        ErrorCategory.CONTENT_FILTER: "Review input content for policy violations",
        ErrorCategory.AUTHENTICATION: "Check API key configuration",
        ErrorCategory.NETWORK_ERROR: "Check network connectivity, retry later",
        ErrorCategory.UNKNOWN: "Investigate error details manually",
    }
    return actions.get(category, "Unknown - investigate manually")


def analyze_errors(
    errors: Sequence[ErrorInfo],
    total_samples: int | None = None,
) -> ErrorAnalysisResult:
    """Analyze a collection of errors.

    Args:
        errors: Sequence of ErrorInfo objects.
        total_samples: Total number of samples (for error rate calculation).

    Returns:
        ErrorAnalysisResult with aggregated statistics.
    """
    errors_list = list(errors)
    n_errors = len(errors_list)

    if total_samples is None:
        total_samples = n_errors

    # Count by category
    by_category: dict[ErrorCategory, int] = Counter()
    for e in errors_list:
        by_category[e.category] += 1

    # Count by severity
    by_severity: dict[ErrorSeverity, int] = Counter()
    for e in errors_list:
        by_severity[e.severity] += 1

    # Count by error type
    by_type: dict[str, int] = Counter()
    for e in errors_list:
        by_type[e.error_type] += 1

    # Find patterns
    patterns = find_error_patterns(errors_list)

    # Build timeline if timestamps available
    timeline = None
    errors_with_time = [e for e in errors_list if e.timestamp]
    if errors_with_time:
        # Group by hour
        from collections import defaultdict
        hourly: dict[datetime, int] = defaultdict(int)
        for e in errors_with_time:
            hour = e.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly[hour] += 1
        timeline = sorted(hourly.items())

    return ErrorAnalysisResult(
        total_errors=n_errors,
        total_samples=total_samples,
        error_rate=n_errors / total_samples if total_samples > 0 else 0.0,
        by_category=dict(by_category),
        by_severity=dict(by_severity),
        by_type=dict(by_type),
        patterns=patterns,
        errors=errors_list,
        timeline=timeline,
    )


def find_error_patterns(
    errors: list[ErrorInfo],
    min_count: int = 2,
) -> list[ErrorPattern]:
    """Identify recurring error patterns.

    Args:
        errors: List of error info objects.
        min_count: Minimum occurrences to consider a pattern.

    Returns:
        List of ErrorPattern objects sorted by frequency.
    """
    # Group by error type
    type_groups: dict[str, list[ErrorInfo]] = {}
    for e in errors:
        if e.error_type not in type_groups:
            type_groups[e.error_type] = []
        type_groups[e.error_type].append(e)

    patterns = []
    for error_type, group in type_groups.items():
        if len(group) >= min_count:
            # Find the category (should be same for all in group)
            category = group[0].category
            suggested_action = get_suggested_action(category)

            patterns.append(ErrorPattern(
                pattern=error_type,
                category=category,
                count=len(group),
                sample_ids=[e.sample_id for e in group],
                example_message=group[0].message[:200],  # Truncate long messages
                suggested_action=suggested_action,
            ))

    # Sort by frequency (descending)
    patterns.sort(key=lambda p: p.count, reverse=True)
    return patterns


def analyze_errors_by_intent(
    errors: Sequence[ErrorInfo],
) -> dict[str | None, ErrorAnalysisResult]:
    """Analyze errors grouped by intent.

    Args:
        errors: Sequence of ErrorInfo objects.

    Returns:
        Dict mapping intent to ErrorAnalysisResult.
    """
    # Group by intent
    intent_groups: dict[str | None, list[ErrorInfo]] = {}
    for e in errors:
        if e.intent not in intent_groups:
            intent_groups[e.intent] = []
        intent_groups[e.intent].append(e)

    # Analyze each group
    results = {}
    for intent, group in intent_groups.items():
        results[intent] = analyze_errors(group)

    return results


def analyze_errors_by_input_length(
    errors: Sequence[ErrorInfo],
    buckets: Sequence[int] = (100, 500, 1000, 2000, 5000),
) -> dict[str, ErrorAnalysisResult]:
    """Analyze errors grouped by input length buckets.

    Args:
        errors: Sequence of ErrorInfo objects.
        buckets: Upper bounds for length buckets.

    Returns:
        Dict mapping bucket label to ErrorAnalysisResult.
    """
    def get_bucket(length: int | None) -> str:
        if length is None:
            return "unknown"
        for upper in buckets:
            if length <= upper:
                return f"<={upper}"
        return f">{buckets[-1]}"

    # Group by bucket
    bucket_groups: dict[str, list[ErrorInfo]] = {}
    for e in errors:
        bucket = get_bucket(e.input_length)
        if bucket not in bucket_groups:
            bucket_groups[bucket] = []
        bucket_groups[bucket].append(e)

    # Analyze each group
    results = {}
    for bucket, group in bucket_groups.items():
        results[bucket] = analyze_errors(group)

    return results


def get_error_summary_for_run(
    session,
    run_id: int,
) -> ErrorAnalysisResult:
    """Get error analysis for a specific run from the database.

    Under the 3-table schema, error information is stored as JSONB in
    the ``error_summary`` column on the Run. This function reads that
    column and returns a lightweight ErrorAnalysisResult.

    Args:
        session: SQLAlchemy session.
        run_id: ID of the run to analyze.

    Returns:
        ErrorAnalysisResult with error statistics.
    """
    from conformal_relevance.db.models import Run

    run = session.get(Run, run_id)
    if not run:
        return ErrorAnalysisResult(
            total_samples=0, total_errors=0, error_rate=0.0,
            success_rate=1.0, by_category={}, by_severity={},
            by_type={}, patterns=[], timeline=None,
        )

    total = (run.n_samples_scored or 0) + (run.n_samples_failed or 0)
    summary = run.error_summary or {}

    # Reconstruct ErrorInfo objects from stored sample entries
    errors: list[ErrorInfo] = []
    for sample_entry in summary.get("samples", []):
        info = categorize_error(
            error_message=sample_entry.get("message", "Unknown error"),
            sample_id=sample_entry.get("index"),
            run_id=run_id,
            retry_count=sample_entry.get("retries", 0),
        )
        errors.append(info)

    return analyze_errors(errors, total)


def get_error_summary_for_dataset(
    session,
    dataset_id: int,
) -> ErrorAnalysisResult:
    """Get error analysis for all runs of a dataset.

    Aggregates ``error_summary`` JSONB from all completed runs of the dataset.

    Args:
        session: SQLAlchemy session.
        dataset_id: ID of the dataset.

    Returns:
        ErrorAnalysisResult with aggregated error statistics.
    """
    from sqlalchemy import select
    from conformal_relevance.db.models import Run

    runs = session.execute(
        select(Run)
        .where(Run.dataset_id == dataset_id)
        .where(Run.status.in_(["completed", "failed"]))
    ).scalars().all()

    total_samples = 0
    all_errors: list[ErrorInfo] = []

    for run in runs:
        total_samples += (run.n_samples_scored or 0) + (run.n_samples_failed or 0)
        summary = run.error_summary or {}
        for sample_entry in summary.get("samples", []):
            info = categorize_error(
                error_message=sample_entry.get("message", "Unknown error"),
                sample_id=sample_entry.get("index"),
                run_id=run.id,
                retry_count=sample_entry.get("retries", 0),
            )
            all_errors.append(info)

    return analyze_errors(all_errors, total_samples)


def format_error_report(analysis: ErrorAnalysisResult) -> str:
    """Format error analysis as a text report.

    Args:
        analysis: ErrorAnalysisResult to format.

    Returns:
        Formatted text report.
    """
    lines = [
        "=" * 60,
        "ERROR ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Total Samples: {analysis.total_samples}",
        f"Total Errors: {analysis.total_errors}",
        f"Error Rate: {analysis.error_rate:.1%}",
        f"Success Rate: {analysis.success_rate:.1%}",
        "",
        "-" * 40,
        "ERRORS BY CATEGORY",
        "-" * 40,
    ]

    for category in ErrorCategory:
        count = analysis.by_category.get(category, 0)
        if count > 0:
            pct = count / analysis.total_errors * 100 if analysis.total_errors > 0 else 0
            lines.append(f"  {category.value}: {count} ({pct:.1f}%)")

    lines.extend([
        "",
        "-" * 40,
        "ERRORS BY SEVERITY",
        "-" * 40,
    ])

    for severity in ErrorSeverity:
        count = analysis.by_severity.get(severity, 0)
        if count > 0:
            pct = count / analysis.total_errors * 100 if analysis.total_errors > 0 else 0
            lines.append(f"  {severity.value}: {count} ({pct:.1f}%)")

    if analysis.patterns:
        lines.extend([
            "",
            "-" * 40,
            "TOP ERROR PATTERNS",
            "-" * 40,
        ])

        for i, pattern in enumerate(analysis.patterns[:5], 1):
            lines.append(f"  {i}. {pattern.pattern} ({pattern.count} occurrences)")
            lines.append(f"     Category: {pattern.category.value}")
            lines.append(f"     Action: {pattern.suggested_action}")
            lines.append("")

    return "\n".join(lines)
