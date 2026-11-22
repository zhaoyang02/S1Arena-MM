"""Bridges between closed-loop suites and the S1Arena-MM decision service.

Each module plays one episode of one suite from inside that suite's own Python environment and imports only the
standard library and the suite's code: `python -m jevarena_mm.bridges.<suite> --endpoint URL --out DIR ...`. Every step
is posted to the service (jevarena_mm/serve.py) as text with <image:k> markers, the step's images and the complete list
of legal actions; the episode's outcome is written to DIR/outcome.json. Started by `jevarena-mm play`.
"""
