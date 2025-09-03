"""
Custom tools for the QA AI Agent system.

This package provides specialized tools for document processing, retrieval,
and RAG (Retrieval-Augmented Generation) operations. The tools are designed
to integrate seamlessly with CrewAI workflows and provide enhanced capabilities
for information processing.

Main Tools
----------
custom_tool
    RAG tool for document retrieval and processing with FAISS integration.

Classes
-------
RetrieverSettings
    Configuration settings for document retrieval behavior.
ChunkingSettings
    Configuration settings for document chunking.
RagToolInput
    Input schema for the RAG tool.
RagTool
    Custom RAG tool for CrewAI integration.

Functions
---------
load_docs
    Load documents from a directory with support for multiple file types.
chunk_docs
    Split documents into smaller chunks for processing.

Examples
--------
>>> from .tools.custom_tool import RagTool, RetrieverSettings
>>> tool = RagTool(retriever_settings=RetrieverSettings(k=5))
>>> result = tool._run("What is the battery capacity of the Galaxy S22?")

Notes
-----
The tools require proper configuration of embedding models and document paths.
Environment variables must be set for Azure OpenAI API access when using
the default embedding model configuration.
"""
