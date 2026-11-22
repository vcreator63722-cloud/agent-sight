from __future__ import annotations

import json

from agent_sight.audio.transcripts import fit_to_budget, paginate, parse_captions, parse_vtt
from agent_sight.models import Transcript, TranscriptSegment

# Shape of YouTube auto-caption VTT: rolling two-line cues plus 10 ms transition cues.
YT_AUTO = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000 align:start position:0%
hello<00:00:00.500><c> everyone</c>

00:00:02.000 --> 00:00:02.010 align:start position:0%
hello everyone

00:00:02.010 --> 00:00:04.000 align:start position:0%
hello everyone
welcome<00:00:02.500><c> to</c><00:00:03.000><c> the</c><00:00:03.400><c> show</c>

00:00:04.000 --> 00:00:04.010
welcome to the show

00:00:04.010 --> 00:00:06.000
welcome to the show
[Music]
"""


def test_youtube_rolling_captions_deduplicated() -> None:
    segs = parse_vtt(YT_AUTO)
    assert [s.text for s in segs] == ["hello everyone", "welcome to the show", "[Music]"]
    assert parse_captions(YT_AUTO, "vtt")[-1].text == "welcome to the show"  # noise dropped


def test_srt() -> None:
    srt = "1\n00:00:01,000 --> 00:00:02,500\n<i>Hi &amp; bye</i>\n\n2\n00:01:00,000 --> 00:01:02,000\nSecond\n"
    segs = parse_captions(srt, "srt")
    assert segs[0].text == "Hi & bye" and segs[0].end == 2.5
    assert segs[1].start == 60


def test_json3() -> None:
    data = {"events": [
        {"tStartMs": 0, "dDurationMs": 1500, "segs": [{"utf8": "Hello "}, {"utf8": "world"}]},
        {"tStartMs": 1500, "dDurationMs": 100, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 2000, "dDurationMs": 1000},
    ]}
    segs = parse_captions(json.dumps(data), "json3")
    assert len(segs) == 1 and segs[0].text == "Hello world" and segs[0].end == 1.5


def _long_transcript(n: int = 400) -> Transcript:
    return Transcript(source="native", segments=[
        TranscriptSegment(start=i * 5, end=i * 5 + 5, text=f"sentence number {i} about topic {i % 7}")
        for i in range(n)
    ])


def test_fit_to_budget_spreads_over_video() -> None:
    excerpts, coverage = fit_to_budget(_long_transcript(), 300)
    assert 0 < coverage < 1
    total = sum(len(e.text) for e in excerpts) / 4
    assert total < 300
    assert excerpts[0].start == 0 and excerpts[-1].start > 1500  # covers the end too


def test_fit_to_budget_small_transcript_complete() -> None:
    _, coverage = fit_to_budget(_long_transcript(5), 1000)
    assert coverage == 1.0


def test_paginate() -> None:
    segs = _long_transcript(100).segments
    page, nxt = paginate(segs, 200)
    assert page and nxt == segs[len(page)].start
    page, nxt = paginate(segs[:3], 10_000)
    assert nxt is None and len(page) == 3
