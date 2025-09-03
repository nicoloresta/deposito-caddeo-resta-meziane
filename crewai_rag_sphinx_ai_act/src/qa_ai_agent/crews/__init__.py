"""
Specialized crews for different types of information processing.

This package contains specialized crews that handle different aspects of
information retrieval and processing in the QA AI Agent system. Each crew
is designed to work with specific types of queries and information sources,
providing optimized processing for different use cases.

Crews
-----
rag_crew
    Handles smartphone-related queries using RAG techniques with pre-indexed
    documents. Optimized for technical specifications and detailed information.

summarizer_crew
    Creates comprehensive summaries of retrieved information, organizing
    content into structured, readable reports suitable for end users.

web_search_crew
    Performs web searches for general topics that fall outside the scope
    of pre-indexed documents. Provides current, up-to-date information.

manager_choice_crew
    Routes user queries to appropriate processing methods based on topic
    analysis. Acts as an intelligent decision-maker in the QA flow.

ethic_checker_crew
    Evaluates query appropriateness and ethical considerations before
    processing. Acts as a safety filter to ensure responsible AI usage.

Architecture
-----------
Each crew follows the CrewAI framework pattern with:
- Specialized agents for specific tasks
- Configurable tasks with YAML-based configuration
- Sequential or hierarchical processing workflows
- Integration with custom tools and external APIs

Examples
--------
>>> from .crews.rag_crew.rag_crew import RagCrew
>>> crew = RagCrew()
>>> result = crew.crew().kickoff(inputs={'topic': 'smartphone features'})

>>> from .crews.web_search_crew.web_search_crew import WebSearchCrew
>>> crew = WebSearchCrew()
>>> result = crew.crew().kickoff(inputs={'prompt': 'latest technology trends'})

Notes
-----
All crews require proper configuration through YAML files in their respective
config directories. The configuration files define agent behaviors, task
parameters, and workflow settings for optimal performance.
"""
