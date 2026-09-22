# examples/smoke

These are **integration smoke tests**, not benchmarks. They drive
`openai.ChatCompletion` against a live API, so their reported latencies
include network round-trips, API throttling, and TLS handshakes. They are
useful for verifying that the OpenAI adapter still works end-to-end after a
change. They are **not** suitable for measuring GPTCache memory or speed.

For benchmarking, use:

    examples/benchmark/benchmark_qqp.py

which runs a 4-cell isolation matrix (encoder x index), reports pure
search latency separately from end-to-end, measures FAISS index RAM
via `faiss.serialize_index`, and is the only harness referenced by
`docs/memory-speed-plan.md`.
