"""Canonical simulator for Phase I.

One implementation only: pure stdlib, so that every audit built on it executes
in the dependency-free CI job rather than skipping itself. A faster numeric
implementation would be a second code path for the same physics, which is the
last place a silent divergence should be allowed to live.
"""
