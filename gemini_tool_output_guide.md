# Gemini Tool Call Output Guide

## Purpose

This guide explains how we need Gemini to format its tool call outputs so Cursor can parse them correctly. Since we're passing content directly through without transformation in `gemini_proxy_v2.py`, we need to ensure the output format matches what Cursor expects.

## The Correct Format For Tool Calls

When we want Gemini to use a tool in Cursor, it should output text in this exact format:

```
<function_calls>
<invoke name="run_terminal_cmd">
<parameter name="command">echo "Hello world" 