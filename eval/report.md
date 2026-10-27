# Idea pipeline check

Run at 2026-10-27T01:11:00+00:00. Model `gpt-4.1-mini`, embeddings `text-embedding-3-small`. Run on a copy of the index with an empty ideas log.

Each round asks for 15 ideas (any format, at most 2 per lesson, passages used within 30 days skipped) and then marks every kept idea as used.

## Rounds

| Round | Proposed | Kept | Grounded | Lessons | From new lessons | Repeats | Dropped | Fresh at start |
|---|---|---|---|---|---|---|---|---|
| 1 | 15 | 15 | 100% | 12 | 15 | 0 | none | 43 |
| 2 | 15 | 15 | 100% | 11 | 0 | 0 | none | 28 |
| 3 | 13 | 13 | 100% | 9 | 0 | 0 | none | 13 |

Grounded = kept / proposed: the share of ideas whose quote was found in the passage by code. Repeats must be 0.

## Topic request: "leftovers"

Asked for 5; the search found 5 passages; kept 5. 4 of the kept ideas come from passages that mention 'leftovers'. Dropped: none.

## Scripts (first ideas of round 1)

| Idea | Format | Words | Warnings |
|---|---|---|---|
| How to Keep Your Partner from Snacking on Batch Cooking Components | post | 129 | none |
| Why the 90-Minute Sunday is Your Batch Cooking Game-Changer | talking_head | 121 | none |
| The Empty-Sink Rule: Keep Your Kitchen Calm | text_graphic | 25 | none |

## Totals

- LLM calls: 7
- Tokens: 11943 in, 7214 out, 2 embedding
- Estimated cost: $0.0163
- Latency per LLM call: mean 7242 ms, max 16741 ms
