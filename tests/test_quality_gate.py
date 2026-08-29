from pathlib import Path

from audio_transcriber.quality_gate import analyze_transcripts_directory, compare_reports


def _write_transcript(path: Path, *, transcript_body: str, chunks: int = 1) -> None:
    path.write_text(
        (
            "# Audio Transcription\n\n"
            "- Source file: sample.mp3\n"
            "- Generated at: 2026-08-29T10:00:00+00:00\n"
            f"- Chunks: {chunks}\n"
            "- Original format: mp3\n"
            "- Working format: mp3\n\n"
            "## Transcript\n\n"
            f"{transcript_body}\n"
        ),
        encoding="utf-8",
    )


def test_analyze_transcripts_directory_collects_metrics(tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    _write_transcript(
        transcripts_dir / "a.md",
        transcript_body="ReelsMaker update\nうん。\nok",
        chunks=2,
    )
    _write_transcript(
        transcripts_dir / "b.md",
        transcript_body="enrichment done\nall good",
        chunks=1,
    )

    report = analyze_transcripts_directory(
        transcripts_dir,
        terms=("ReelsMaker", "enrichment"),
        top=2,
    )

    assert report["transcript_count"] == 2
    assert report["cjk_lines"] == 1
    assert report["term_hits"]["ReelsMaker"] == 1
    assert report["term_hits"]["enrichment"] == 1
    assert len(report["top_worst_cjk"]) == 2


def test_compare_reports_passes_when_candidate_improves_metrics() -> None:
    baseline = {
        "cjk_lines": 10,
        "short_noise_lines": 8,
        "term_hits": {"ReelsMaker": 2, "enrichment": 1},
    }
    candidate = {
        "cjk_lines": 0,
        "short_noise_lines": 4,
        "term_hits": {"ReelsMaker": 2, "enrichment": 1},
    }

    passed, details = compare_reports(baseline, candidate, min_cjk_reduction=90.0)

    assert passed is True
    assert details["cjk_reduction_pct"] == 100.0
    assert details["term_regressions"] == []


def test_compare_reports_fails_on_term_regression() -> None:
    baseline = {
        "cjk_lines": 10,
        "short_noise_lines": 8,
        "term_hits": {"ReelsMaker": 2, "enrichment": 1},
    }
    candidate = {
        "cjk_lines": 1,
        "short_noise_lines": 4,
        "term_hits": {"ReelsMaker": 1, "enrichment": 1},
    }

    passed, details = compare_reports(baseline, candidate, min_cjk_reduction=80.0)

    assert passed is False
    assert details["term_regressions"] == ["ReelsMaker"]
