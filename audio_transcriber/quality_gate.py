from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

DEFAULT_TERMS = (
    "ReelsMaker",
    "enrichment",
    "UV sync",
    "For You",
    "scheduler",
    "ContentLoader",
    "media segments",
)
DEFAULT_MIN_CJK_REDUCTION = 90.0
TRANSCRIPT_MARKER = "## Transcript"
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
TERM_LINE_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class FileMetrics:
    file_name: str
    source_file: str | None
    chunks: int
    lines: int
    cjk_lines: int
    short_noise_lines: int
    term_hits: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "source_file": self.source_file,
            "chunks": self.chunks,
            "lines": self.lines,
            "cjk_lines": self.cjk_lines,
            "cjk_rate": _safe_rate(self.cjk_lines, self.lines),
            "short_noise_lines": self.short_noise_lines,
            "short_noise_rate": _safe_rate(self.short_noise_lines, self.lines),
            "term_hits": self.term_hits,
        }


def analyze_transcripts_directory(
    input_dir: str | Path,
    *,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    top: int = 5,
) -> dict[str, Any]:
    directory = Path(input_dir)
    files = sorted(directory.glob("*.md"))
    metrics = [_analyze_markdown_file(path, terms=terms) for path in files]

    total_lines = sum(item.lines for item in metrics)
    total_cjk_lines = sum(item.cjk_lines for item in metrics)
    total_short_noise = sum(item.short_noise_lines for item in metrics)
    term_totals = {term: 0 for term in terms}
    for file_metrics in metrics:
        for term, count in file_metrics.term_hits.items():
            term_totals[term] += count

    top_worst = sorted(metrics, key=lambda item: item.cjk_lines, reverse=True)[: max(0, top)]
    return {
        "transcript_count": len(metrics),
        "total_lines": total_lines,
        "cjk_lines": total_cjk_lines,
        "cjk_rate": _safe_rate(total_cjk_lines, total_lines),
        "short_noise_lines": total_short_noise,
        "short_noise_rate": _safe_rate(total_short_noise, total_lines),
        "term_hits": term_totals,
        "top_worst_cjk": [item.to_dict() for item in top_worst],
    }


def compare_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    min_cjk_reduction: float = DEFAULT_MIN_CJK_REDUCTION,
) -> tuple[bool, dict[str, Any]]:
    baseline_cjk = int(baseline_report.get("cjk_lines", 0))
    candidate_cjk = int(candidate_report.get("cjk_lines", 0))
    cjk_reduction_pct = _relative_reduction(baseline_cjk, candidate_cjk)
    cjk_pass = cjk_reduction_pct >= min_cjk_reduction

    baseline_noise = int(baseline_report.get("short_noise_lines", 0))
    candidate_noise = int(candidate_report.get("short_noise_lines", 0))
    short_noise_pass = candidate_noise <= baseline_noise

    baseline_terms = baseline_report.get("term_hits", {})
    candidate_terms = candidate_report.get("term_hits", {})
    term_regressions = sorted(
        term
        for term, baseline_count in baseline_terms.items()
        if int(candidate_terms.get(term, 0)) < int(baseline_count)
    )
    terms_pass = not term_regressions

    passed = cjk_pass and short_noise_pass and terms_pass
    details = {
        "passed": passed,
        "min_cjk_reduction": min_cjk_reduction,
        "cjk_reduction_pct": cjk_reduction_pct,
        "baseline_cjk_lines": baseline_cjk,
        "candidate_cjk_lines": candidate_cjk,
        "baseline_short_noise_lines": baseline_noise,
        "candidate_short_noise_lines": candidate_noise,
        "term_regressions": term_regressions,
    }
    return passed, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcription quality baseline and gate utility.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Build baseline report from transcripts.")
    baseline_parser.add_argument("--input-dir", required=True, help="Directory with markdown transcripts.")
    baseline_parser.add_argument(
        "--terms",
        default=",".join(DEFAULT_TERMS),
        help="Comma-separated domain terms to track.",
    )
    baseline_parser.add_argument("--top", type=int, default=5, help="How many worst files to include.")
    baseline_parser.add_argument("--output", help="Optional output path for JSON report.")

    compare_parser = subparsers.add_parser("compare", help="Compare baseline and candidate JSON reports.")
    compare_parser.add_argument("--baseline", required=True, help="Path to baseline report JSON.")
    compare_parser.add_argument("--candidate", required=True, help="Path to candidate report JSON.")
    compare_parser.add_argument(
        "--min-cjk-reduction",
        type=float,
        default=DEFAULT_MIN_CJK_REDUCTION,
        help="Required relative CJK reduction percentage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "baseline":
        terms = _parse_terms(args.terms)
        report = analyze_transcripts_directory(args.input_dir, terms=terms, top=args.top)
        output_text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
        print(output_text)
        return 0

    if args.command == "compare":
        baseline_report = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        candidate_report = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        passed, details = compare_reports(
            baseline_report,
            candidate_report,
            min_cjk_reduction=args.min_cjk_reduction,
        )
        print(json.dumps(details, ensure_ascii=False, indent=2))
        return 0 if passed else 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _analyze_markdown_file(path: Path, *, terms: tuple[str, ...]) -> FileMetrics:
    content = path.read_text(encoding="utf-8", errors="replace")
    body = _extract_transcript_body(content)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    cjk_lines = sum(1 for line in lines if CJK_RE.search(line))
    short_noise = sum(1 for line in lines if _is_short_noise_line(line))
    term_hits = {
        term: len(re.findall(re.escape(term), body, flags=re.IGNORECASE))
        for term in terms
    }
    return FileMetrics(
        file_name=path.name,
        source_file=_extract_source_file(content),
        chunks=_extract_chunk_count(content),
        lines=len(lines),
        cjk_lines=cjk_lines,
        short_noise_lines=short_noise,
        term_hits=term_hits,
    )


def _extract_transcript_body(content: str) -> str:
    marker_index = content.find(TRANSCRIPT_MARKER)
    if marker_index == -1:
        return content
    return content[marker_index + len(TRANSCRIPT_MARKER) :].strip()


def _extract_source_file(content: str) -> str | None:
    match = re.search(r"- Source file:\s*(.+)", content)
    return match.group(1).strip() if match else None


def _extract_chunk_count(content: str) -> int:
    match = re.search(r"- Chunks:\s*(\d+)", content)
    if not match:
        return 0
    return int(match.group(1))


def _is_short_noise_line(line: str) -> bool:
    if len(line) > 12:
        return False
    if CYRILLIC_RE.search(line):
        return False
    token_count = len(TERM_LINE_TOKEN_RE.findall(line))
    return token_count <= 3


def _parse_terms(raw_terms: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw_terms.split(",") if part.strip())
    return values or DEFAULT_TERMS


def _safe_rate(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((part / total) * 100, 4)


def _relative_reduction(baseline: int, candidate: int) -> float:
    if baseline <= 0:
        return 100.0 if candidate <= 0 else 0.0
    return round(((baseline - candidate) / baseline) * 100, 4)


if __name__ == "__main__":
    raise SystemExit(main())
