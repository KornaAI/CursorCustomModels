#!/usr/bin/env python3
"""
Tool Prompt Generator for Gemini

This script generates a system prompt for Gemini that includes instructions
on how to properly format tool calls for Cursor integration.
"""

import os
import time

def create_system_prompt(base_prompt=""):
    """
    Create a system prompt with tool calling instructions
    
    Args:
        base_prompt: The base system prompt to extend
        
    Returns:
        Full system prompt with tool calling instructions
    """
    tool_instructions = """
# Tool Usage Instructions

You have access to the following tools that you can use to assist the user.
When you need to use a tool, you MUST format your request as follows:

<function_calls>
<invoke name="TOOL_NAME">
<parameter name="PARAM_NAME">PARAM_VALUE 