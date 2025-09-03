"""
RAG (Retrieval-Augmented Generation) crew for smartphone information retrieval.

This module implements a specialized crew for handling smartphone-related queries
using RAG techniques. It combines prompt rewriting and document retrieval to
provide accurate, context-aware responses based on pre-indexed smartphone documents.

Classes
-------
RagCrew
    Crew class that orchestrates RAG operations for smartphone queries.

Examples
--------
>>> from .crews.rag_crew.rag_crew import RagCrew
>>> crew = RagCrew()
>>> result = crew.crew().kickoff(inputs={'topic': 'battery capacity'})
"""

from pathlib import Path
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from ...tools.qdrant_custom_tool import RagTool


@CrewBase
class RagCrew:
    """
    RAG crew for smartphone information retrieval.

    This crew specializes in handling queries related to smartphones by using
    Retrieval-Augmented Generation techniques. It combines two main agents:
    a prompt rewriter that optimizes user queries for better retrieval, and
    a retriever that searches through pre-indexed smartphone documents.

    The crew is designed to work with the smartphone document collection and
    provides relevant, accurate information based on the indexed content.

    Attributes
    ----------
    agents : List[BaseAgent]
        List of agents managed by the crew (automatically populated by decorators).
    tasks : List[Task]
        List of tasks managed by the crew (automatically populated by decorators).

    Examples
    --------
    >>> crew = RagCrew()
    >>> result = crew.crew().kickoff(inputs={'topic': 'Galaxy S22 specifications'})
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_prompt_rewriter(self) -> Agent:
        """
        Create the RAG prompt rewriter agent.

        This agent is responsible for taking user queries and rewriting them
        to be more effective for document retrieval. It may expand abbreviations,
        add synonyms, or restructure the query to better match the indexed content.

        Returns
        -------
        Agent
            Configured agent for prompt rewriting with settings from the
            agents configuration file.

        Notes
        -----
        The agent configuration is loaded from the 'rag_prompt_rewriter' section
        of the agents configuration file.
        """
        return Agent(
            config=self.agents_config["rag_prompt_rewriter"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def rag_retriever(self) -> Agent:
        """
        Create the RAG retriever agent.

        This agent is responsible for searching through the indexed smartphone
        documents to find the most relevant information for the user's query.
        It uses the RagTool to perform semantic search and retrieve appropriate
        document chunks.

        Returns
        -------
        Agent
            Configured agent for document retrieval with RagTool integration
            and settings from the agents configuration file.

        Notes
        -----
        The agent is equipped with the RagTool for document retrieval and
        configuration is loaded from the 'rag_retriever' section of the
        agents configuration file.
        """
        return Agent(
            config=self.agents_config["rag_retriever"],  # type: ignore[index]
            verbose=True,
            tools=[RagTool()],
        )

    @task
    def rag_prompt_task(self) -> Task:
        """
        Create the RAG prompt rewriting task.

        This task defines the work to be performed by the prompt rewriter agent.
        It specifies how user queries should be processed and optimized for
        better retrieval performance.

        Returns
        -------
        Task
            Configured task for prompt rewriting with settings from the
            tasks configuration file.

        Notes
        -----
        Task configuration is loaded from the 'rag_prompt_task' section of
        the tasks configuration file.
        """
        return Task(
            config=self.tasks_config["rag_prompt_task"],  # type: ignore[index]
        )

    @task
    def rag_retrieval_task(self) -> Task:
        """
        Create the RAG retrieval task.

        This task defines the work to be performed by the retriever agent.
        It specifies how the rewritten query should be used to search through
        the indexed documents and retrieve relevant information.

        Returns
        -------
        Task
            Configured task for document retrieval with settings from the
            tasks configuration file.

        Notes
        -----
        Task configuration is loaded from the 'rag_retrieval_task' section of
        the tasks configuration file.
        """
        return Task(
            config=self.tasks_config["rag_retrieval_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """
        Create the orchestrated RAG crew.

        This method assembles the agents and tasks into a working crew that
        processes queries sequentially: first rewriting the prompt, then
        retrieving relevant documents.

        Returns
        -------
        Crew
            Configured crew with sequential processing, combining the prompt
            rewriter and retriever agents with their respective tasks.

        Notes
        -----
        The crew uses sequential processing to ensure that prompt rewriting
        is completed before document retrieval begins. This allows the
        retriever to work with an optimized query for better results.
        """
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
