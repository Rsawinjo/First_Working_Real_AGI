"""
Tool Registry for Autonomous AGI System

This module provides a registry of tools that the AGI can use autonomously
to solve problems and gather information. Tools are callable functions or
APIs that extend the system's capabilities beyond its core learning engine.
"""

import requests
import json
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

# Load settings
try:
    from ..config.settings import WOLFRAM_ALPHA_APP_ID
except ImportError:
    WOLFRAM_ALPHA_APP_ID = ""


class Tool(ABC):
    """Abstract base class for tools."""
    @abstractmethod
    def execute(self, query: str) -> str:
        """Execute the tool with a query and return the result."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool's name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""


class CalculatorTool(Tool):
    """Simple calculator tool using eval (for basic math)."""
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Performs basic mathematical calculations."

    def execute(self, query: str) -> str:
        try:
            # Sanitize input to prevent code injection
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in query):
                return "Error: Invalid characters in calculation."
            result = eval(query)  # Note: eval is used for simplicity; consider safer alternatives in production
            return str(result)
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            return f"Error: {str(e)}"


class WebAPITool(Tool):
    """Generic web API tool for making HTTP requests."""
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "web_api"

    @property
    def description(self) -> str:
        return "Makes HTTP requests to web APIs for data retrieval."

    def execute(self, query: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.get(f"{self.base_url}{query}", headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"Error: {str(e)}"


class WolframAlphaTool(Tool):
    """Tool for querying Wolfram Alpha (requires API key)."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "wolfram_alpha"

    @property
    def description(self) -> str:
        return "Queries Wolfram Alpha for mathematical and scientific computations."

    def execute(self, query: str) -> str:
        try:
            url = f"http://api.wolframalpha.com/v2/query?input={query}&appid={self.api_key}&output=json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Extract plain text result
            pods = data.get("queryresult", {}).get("pods", [])
            if pods:
                return pods[0].get("subpods", [{}])[0].get("plaintext", "No result found.")
            return "No result found."
        except requests.RequestException as e:
            return f"Error: {str(e)}"


class ToolRegistry:
    """Registry to manage and select tools."""
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in tools."""
        self.register_tool(CalculatorTool())
        # Register Wolfram Alpha if API key is available
        if WOLFRAM_ALPHA_APP_ID:
            self.register_tool(WolframAlphaTool(WOLFRAM_ALPHA_APP_ID))
        # Note: Add other tools with API keys here

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": tool.name, "description": tool.description} for tool in self.tools.values()]

    def select_tool(self, query: str, llm_interface) -> Optional[str]:
        """
        Use LLM to select the best tool for a query.
        Returns the tool name or None if no tool fits.
        """
        tools_list = self.list_tools()
        if not tools_list:
            return None

        prompt = f"""
        Given the query: "{query}"
        Available tools: {json.dumps(tools_list, indent=2)}
        Which tool should be used? Respond with only the tool name, or "none" if no tool fits.
        """
        response = llm_interface.generate_response(prompt, max_length=50).strip().lower()
        if response in self.tools:
            return response
        return None

    def execute_tool(self, tool_name: str, query: str) -> str:
        tool = self.get_tool(tool_name)
        if tool:
            return tool.execute(query)
        return "Tool not found."


# Global registry instance
tool_registry = ToolRegistry()