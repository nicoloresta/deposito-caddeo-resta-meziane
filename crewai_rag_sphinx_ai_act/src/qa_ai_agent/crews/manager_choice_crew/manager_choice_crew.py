"""
Manager choice crew for routing user queries to appropriate processing methods.

This module implements a specialized crew that determines whether a user's
query should be handled by the smartphone RAG system or routed to general
web search. It acts as a decision-making component in the QA flow.

The crew analyzes the user's topic and makes intelligent routing decisions
to ensure queries are processed by the most appropriate information retrieval
method.

Classes
-------
ManagerChoiceCrew
    Crew class that orchestrates query routing decisions.

Examples
--------
>>> from .crews.manager_choice_crew.manager_choice_crew import ManagerChoiceCrew
>>> crew = ManagerChoiceCrew()
>>> result = crew.manager_choice_crew().kickoff(inputs={'topic': 'smartphone specifications'})
"""

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class ManagerChoiceCrew:
    """
    Manager choice crew for routing user queries to appropriate processing methods.

    This crew is responsible for analyzing user topics and determining the most
    appropriate processing path. It evaluates whether a query is related to
    smartphones (which can benefit from the specialized RAG system) or should
    be handled by general web search.

    The crew acts as an intelligent router in the QA flow, ensuring that
    each query is processed by the most suitable information retrieval method
    for optimal results.

    Attributes
    ----------
    agents : List[BaseAgent]
        List of agents managed by the crew (automatically populated by decorators).
    tasks : List[Task]
        List of tasks managed by the crew (automatically populated by decorators).

    Examples
    --------
    >>> crew = ManagerChoiceCrew()
    >>> result = crew.manager_choice_crew().kickoff(inputs={'topic': 'latest technology news'})
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def manager_choice(self) -> Agent:
        """
        Create the manager choice agent.

        This agent is responsible for analyzing the user's topic and making
        routing decisions. It evaluates the content and context of the query
        to determine whether it should be processed by the smartphone RAG
        system or routed to general web search.

        Returns
        -------
        Agent
            Configured agent for topic analysis and routing decisions with
            settings from the agents configuration file.

        Notes
        -----
        The agent configuration is loaded from the 'manager_choice' section of
        the agents configuration file.
        """
        return Agent(config=self.agents_config["manager_choice"], verbose=True)

    @task
    def manager_choice_task(self) -> Task:
        """
        Create the manager choice task.

        This task defines the work to be performed by the manager choice agent.
        It specifies how user topics should be analyzed and what criteria
        should be used to make routing decisions.

        Returns
        -------
        Task
            Configured task for topic analysis and routing with settings
            from the tasks configuration file.

        Notes
        -----
        Task configuration is loaded from the 'manager_choice_task' section of
        the tasks configuration file.
        """
        return Task(
            config=self.tasks_config["manager_choice_task"],
        )

    @crew
    def manager_choice_crew(self) -> Crew:
        """
        Create the orchestrated manager choice crew.

        This method assembles the manager choice agent and task into a working
        crew that analyzes topics and makes routing decisions. The crew uses
        sequential processing to ensure the analysis is completed before
        returning the routing decision.

        Returns
        -------
        Crew
            Configured crew with sequential processing, combining the
            manager choice agent with the manager choice task.

        Notes
        -----
        The crew is designed to work as a decision-making component in the
        QA flow, determining the optimal processing path for each user query.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
