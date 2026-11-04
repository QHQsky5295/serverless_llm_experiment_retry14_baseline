The first read-only analysis exited before writing retained.json: its config
lookup incorrectly used `coordination` instead of the existing
`resource_coordination` key. KeyError occurred at line 40, after source parsing,
before any demand classification or result write. Scope
primelora-tc-test-c8254de8c013445e830f2ff5d302bdd7.scope exited 1.
Only the lookup key was corrected. No service/config/input mutation or GPU run.
