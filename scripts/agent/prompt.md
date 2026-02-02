You are running in the keiba repository. Follow AGENTS.md and memory.md.

Hypothesis ID: {{id}}
Title: {{title}}
Hypothesis: {{hypothesis}}
Risk level: {{risk_level}}
Max diff size: {{max_diff_size}}
Change scope:
{{change_scope}}
Acceptance criteria:
{{acceptance_criteria}}
Metrics:
{{metrics}}

Instructions:
- Implement the smallest change needed to test the hypothesis.
- Keep total diff <= {{max_diff_size}} lines (added + deleted).
- Update or add tests/eval as needed.
- Update docs/experiments/{{id}}.md using docs/experiments/_template.md and fill required fields.
- Do not run destructive commands or full-privilege flags.
- If you run any commands, include them in the JSON output.

Output:
- Return ONLY valid JSON matching scripts/agent/output_schema.json.
- Include a PR-ready body in `pr_body` (Markdown).
