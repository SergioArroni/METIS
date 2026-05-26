"""Integration tests — real component interactions with minimal mocking.

WHY: Unit tests verify each piece in isolation. Integration tests verify
that components actually work TOGETHER: data flows correctly through the
pipeline, types are compatible, and real aggregation produces sensible results.

These tests use programmatically generated DataFrames (no disk I/O).
"""
