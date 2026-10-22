# Answer-evaluation benchmark

`answer_benchmark.csv` is a 132-case development benchmark for checking the
evaluator against varied answer styles. It contains:

- `project_sample` rows copied from the bundled sample answer sheets.
- `curated_variant` paraphrases and fuller/partial answers.
- `curated_negative` incorrect or contradictory answers.
- A 30-case `development` split based on the Deep Learning answer-key sample.
- A 102-case `test` split covering Artificial Intelligence, Natural Language
  Processing, Machine Learning, Compiler Design, and Computer Networks.

Labels are editorially assigned for development; they have **not** been
independently verified by teachers or graders. They are not ground-truth
human grades and must not be used to claim real-world accuracy. The held-out
topic split reduces direct answer-topic overlap, but both sets are small and
come from the same bundled educational sample material.

`acceptable` is the positive class. `partial` and `incorrect` are treated as
manual-review outcomes for the benchmark's binary pass/review metrics. This
is a simplification for analysis, not a universal grading policy.
The benchmark uses a 37.5% pass/review threshold, selected on the development
split for the highest development accuracy. The held-out test split is not
used to select it. This threshold changes only binary benchmark predictions,
not awarded marks.

The benchmark is currently intentionally run with lexical concept coverage
only. Semantic-model metrics should be reported separately after installing
the optional dependencies and calibrating the semantic decision rule on the
development split. Do not silently mix lexical-only and semantic results.

To make the benchmark suitable for a genuine evaluation, have at least two
independent graders label a broader, privacy-reviewed collection of actual
student answers, adjudicate disagreements, and replace or separately add
those adjudicated examples with a documented source and split.
