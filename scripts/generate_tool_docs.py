#!/usr/bin/env python3
"""
This script generates documentation with the correct tool call format
that Gemini should output in order for Cursor to parse it correctly.
"""

import os

# Define the documentation content
DOCUMENTATION = """# Gemini Tool Output Format for Cursor

## Overview

This document explains how Gemini needs to format its responses when using tools via `gemini_proxy_v2.py`.
Since we're passing content directly without transformation, Gemini must output in a format Cursor understands.

## System Prompt Template

Add this to your system prompt when configuring Gemini:

```
When you need to use a tool, format your request as follows:

<function_calls>
<invoke name="TOOL_NAME">
<parameter name="PARAM_NAME">PARAM_VALUE 