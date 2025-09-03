"""
Ethics checker crew for evaluating query appropriateness and ethical considerations.

This module implements a specialized crew that analyzes user queries to determine
whether they are appropriate and ethical to process. It acts as a safety filter
in the QA flow, ensuring that only suitable topics are processed.

The crew evaluates the ethical implications of user requests and provides
reasoning for its decisions, helping maintain responsible AI usage.

Classes
-------
EthicAnalysis
    Pydantic model for ethics check results.
EthicCheckerCrew
    Crew class that orchestrates ethics checking operations.

Examples
--------
>>> from .crews.ethic_checker_crew.ethic_checker_crew import EthicCheckerCrew
>>> crew = EthicCheckerCrew()
>>> result = crew.crew().kickoff(inputs={'topic': 'smartphone specifications'})
"""

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field


class EthicAnalysis(BaseModel):
    """
    Pydantic model for the output of the ethics check.

    This model defines the structure of the ethics analysis results,
    providing a clear format for the ethical evaluation of user queries.

    Parameters
    ----------
    is_ethical : bool
        True if the topic is ethical and appropriate to process, False otherwise.
        This field determines whether the query can proceed in the QA flow.
    reason : str
        A brief explanation for the ethical judgment, providing transparency
        about why a topic was approved or rejected.

    Examples
    --------
    >>> analysis = EthicAnalysis(is_ethical=True, reason="Topic is about technology specifications")
    >>> analysis.is_ethical
    True
    >>> analysis.reason
    'Topic is about technology specifications'
    """

    is_ethical: bool = Field(
        description="True if the topic is ethical, False otherwise."
    )
    reason: str = Field(description="A brief explanation for the ethical judgment.")


@CrewBase
class EthicCheckerCrew:
    """
    Ethics checker crew for evaluating query appropriateness and ethical considerations.

    This crew is responsible for analyzing user topics to determine whether they
    are appropriate and ethical to process. It acts as a safety filter at the
    beginning of the QA flow, ensuring that only suitable topics proceed to
    information retrieval.

    The crew uses a specialized ethics checking agent to evaluate queries based
    on ethical guidelines and appropriateness criteria. It provides clear reasoning
    for its decisions, promoting transparency and responsible AI usage.

    Attributes
    ----------
    agents : List[BaseAgent]
        List of agents managed by the crew (automatically populated by decorators).
    tasks : List[Task]
        List of tasks managed by the crew (automatically populated by decorators).

    Examples
    --------
    >>> crew = EthicCheckerCrew()
    >>> result = crew.crew().kickoff(inputs={'topic': 'smartphone features'})
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def ethic_checker(self) -> Agent:
        """
        Create the ethics checker agent.

        This agent is responsible for analyzing user topics and determining
        their ethical appropriateness. It evaluates queries based on ethical
        guidelines and provides reasoning for its decisions.

        The agent is configured to return results in the EthicAnalysis format,
        ensuring consistent and structured output for the ethics evaluation.

        Returns
        -------
        Agent
            Configured agent for ethics checking with EthicAnalysis response
            format and settings from the agents configuration file.

        Notes
        -----
        The agent configuration is loaded from the 'ethic_checker' section of
        the agents configuration file. The agent is configured to return
        structured output using the EthicAnalysis model.
        """
        return Agent(
            config=self.agents_config["ethic_checker"],  # type: ignore[index]
            verbose=True,
            response_format=EthicAnalysis,
        )

    @task
    def check_user_topic_ethic(self) -> Task:
        """
        Create the ethics checking task.

        This task defines the work to be performed by the ethics checker agent.
        It specifies how user topics should be evaluated for ethical appropriateness
        and what criteria should be used for the evaluation.

        The task is configured to output results in JSON format using the
        EthicAnalysis model, ensuring consistent and parseable output.

        Returns
        -------
        Task
            Configured task for ethics checking with EthicAnalysis output format
            and settings from the tasks configuration file.

        Notes
        -----
        Task configuration is loaded from the 'check_user_topic_ethic' section of
        the tasks configuration file. The task outputs results in JSON format
        using the EthicAnalysis model for easy parsing and integration.
        """
        return Task(
            config=self.tasks_config["check_user_topic_ethic"],  # type: ignore[index]
            output_json=EthicAnalysis,
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """
        Create the orchestrated ethics checker crew.

        This method assembles the ethics checker agent and task into a working
        crew that evaluates user topics for ethical appropriateness. The crew
        uses sequential processing to ensure the ethics analysis is completed
        before proceeding with the evaluation results.

        Returns
        -------
        Crew
            Configured crew with sequential processing, combining the ethics
            checker agent with the ethics checking task.

        Notes
        -----
        The crew is designed to work as the first safety filter in the QA flow,
        evaluating topics before they proceed to information retrieval. It
        ensures responsible AI usage by filtering out inappropriate or unethical
        queries early in the process.
        """
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
