"""
QA AI Agent system for intelligent information retrieval and processing.

This package implements a comprehensive question-answering system that combines
multiple AI agents working together to provide accurate, relevant information
to user queries. The system features ethical checking, intelligent routing,
and multiple information retrieval methods including RAG and web search.

The system is built on CrewAI's flow framework and provides a modular,
extensible architecture for AI-powered information processing.

Main Components
--------------
main
    Main flow orchestration and user interaction.
tools
    Custom tools for document processing and RAG operations.
crews
    Specialized crews for different types of information processing.

Crews
-----
rag_crew
    Handles smartphone-related queries using RAG techniques.
summarizer_crew
    Creates comprehensive summaries of retrieved information.
web_search_crew
    Performs web searches for general topics.
manager_choice_crew
    Routes queries to appropriate processing methods.
ethic_checker_crew
    Evaluates query appropriateness and ethical considerations.

Tools
-----
custom_tool
    RAG tool for document retrieval and processing.

Examples
--------
>>> from .main import kickoff
>>> kickoff()
What topic would you like to search?
> smartphone specifications
Checking ethical considerations...
Ethics analysis passed.
Managing user choice...
...

Notes
-----
The system requires proper configuration of agents and tasks through YAML
configuration files, and appropriate environment variables for API access.
"""

__version__ = "1.0.0"
__author__ = "QA AI Agent Team"
__description__ = "Intelligent question-answering system with ethical checking and multi-source information retrieval"
