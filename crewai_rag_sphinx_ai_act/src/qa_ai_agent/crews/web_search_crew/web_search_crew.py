"""
Web search crew for general information retrieval.

This module implements a specialized crew for performing web searches on topics
that are not covered by the smartphone RAG system. It combines prompt rewriting
with web search capabilities to find relevant information from the internet.

The crew uses the SerperDev tool for web search operations and can handle
a wide variety of topics beyond the scope of the pre-indexed documents.

Classes
-------
WebSearchCrew
    Crew class that orchestrates web search operations.

Examples
--------
>>> from .crews.web_search_crew.web_search_crew import WebSearchCrew
>>> crew = WebSearchCrew()
>>> result = crew.crew().kickoff(inputs={'prompt': 'latest technology trends'})
"""

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class WebSearchCrew:
    """
    Web search crew for general information retrieval.

    This crew is designed to handle queries that fall outside the scope of
    the smartphone RAG system. It performs web searches to find current,
    up-to-date information on a wide variety of topics.

    The crew combines two main components: a prompt rewriter that optimizes
    search queries, and a web searcher that uses the SerperDev tool to
    find relevant information from the internet.

    Attributes
    ----------
    agents : List[BaseAgent]
        List of agents managed by the crew (automatically populated by decorators).
    tasks : List[Task]
        List of tasks managed by the crew (automatically populated by decorators).

    Examples
    --------
    >>> crew = WebSearchCrew()
    >>> result = crew.crew().kickoff(inputs={'prompt': 'latest smartphone releases'})
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def web_prompt_rewriter(self) -> Agent:
        """
        Create the web prompt rewriter agent.

        This agent is responsible for taking user queries and rewriting them
        to be more effective for web search. It may expand abbreviations,
        add relevant keywords, or restructure the query to improve search
        engine results.

        Returns
        -------
        Agent
            Configured agent for prompt rewriting with settings from the
            agents configuration file.

        Notes
        -----
        The agent configuration is loaded from the 'web_prompt_rewriter' section
        of the agents configuration file.
        """
        return Agent(config=self.agents_config["web_prompt_rewriter"], verbose=True)

    @agent
    def web_searcher(self) -> Agent:
        """
        Create the web searcher agent.

        This agent is responsible for performing web searches using the
        SerperDev tool. It takes the rewritten query and searches the
        internet for relevant, current information on the topic.

        Returns
        -------
        Agent
            Configured agent for web search with SerperDev tool integration
            and settings from the agents configuration file.

        Notes
        -----
        The agent is equipped with the SerperDev tool configured to return
        up to 3 search results. Configuration is loaded from the
        'web_searcher' section of the agents configuration file.
        """
        return Agent(
            config=self.agents_config["web_searcher"],
            verbose=True,
            tools=[SerperDevTool(n_results=3)],
        )

    @task
    def web_prompt_task(self) -> Task:
        """
        Create the web prompt rewriting task.

        This task defines the work to be performed by the prompt rewriter agent.
        It specifies how user queries should be processed and optimized for
        better web search performance.

        Returns
        -------
        Task
            Configured task for prompt rewriting with settings from the
            tasks configuration file.

        Notes
        -----
        Task configuration is loaded from the 'web_prompt_task' section of
        the tasks configuration file.
        """
        return Task(config=self.tasks_config["web_prompt_task"])

    @task
    def web_search_task(self) -> Task:
        """
        Create the web search task.

        This task defines the work to be performed by the web searcher agent.
        It specifies how the rewritten query should be used to search the
        internet and retrieve relevant information.

        Returns
        -------
        Task
            Configured task for web search with settings from the
            tasks configuration file.

        Notes
        -----
        Task configuration is loaded from the 'web_search_task' section of
        the tasks configuration file.
        """
        return Task(config=self.tasks_config["web_search_task"])

    @crew
    def crew(self) -> Crew:
        """
        Create the orchestrated web search crew.

        This method assembles the prompt rewriter and web searcher agents
        with their respective tasks into a working crew. The crew processes
        queries sequentially: first rewriting the prompt, then performing
        the web search.

        Returns
        -------
        Crew
            Configured crew with sequential processing, combining the prompt
            rewriter and web searcher agents with their respective tasks.

        Notes
        -----
        The crew uses sequential processing to ensure that prompt rewriting
        is completed before web search begins. This allows the searcher to
        work with an optimized query for better search results.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
