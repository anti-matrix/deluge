This project builds the simulation specified in docs/SPECS.md.
Build one round at a time (SPECS section 13). Never start the next round without my explicit approval.
Use only the project's virtual environment; never install packages globally.
Verify Mesa API names against the installed version before using them; where the installed library and the spec disagree, the library wins and you report the deviation.
No code may inject myth content, motifs or story vocabulary (SPECS section 11).
All tunable numbers go in configs/, never as literals in code.
If a mechanic seems to need content or a decision the spec doesn't make, stop and ask.
End every round with tests passing, a local git commit (no pushing, no branches), and a report: built, tests, deviations from spec, open problems, next round.
Commit only the files you created or changed in this round (git add with explicit paths, never git add -A). If you find other uncommitted changes, tell me instead of committing them.