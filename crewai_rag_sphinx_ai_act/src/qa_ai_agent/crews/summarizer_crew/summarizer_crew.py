"""
Summarization crew for processing and summarizing information.

This module implements a specialized crew for creating concise, well-structured
summaries of information retrieved from various sources. It takes the output
from RAG or web search operations and produces a comprehensive summary that
is saved to a markdown file for easy review.

Classes
-------
SummarizationCrew
    Crew class that orchestrates summarization operations.

Examples
--------
>>> from .crews.summarizer_crew.summarizer_crew import SummarizationCrew
>>> crew = SummarizationCrew()
>>> result = crew.summarization_crew().kickoff(inputs={'prompt': 'content to summarize'})
"""

from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class SummarizationCrew:
    """
    Summarization crew for processing and summarizing information.

    This crew is responsible for taking the output from other crews (RAG or
    web search) and creating a comprehensive, well-structured summary. It
    uses a specialized summarizer agent to process the content and produce
    a final report that is saved to the output directory.

    The crew is designed to work with various types of content and can
    handle both technical specifications and general information, adapting
    the summary style and format accordingly.

    Attributes
    ----------
    agents : List[BaseAgent]
        List of agents managed by the crew (automatically populated by decorators).
    tasks : List[Task]
        List of tasks managed by the crew (automatically populated by decorators).

    Examples
    --------
    >>> crew = SummarizationCrew()
    >>> result = crew.summarization_crew().kickoff(
    ...     inputs={'prompt': 'content to summarize', 'topic': 'smartphones'}
    ... )
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def summarizer(self) -> Agent:
        """
        Create the summarizer agent.

        This agent is responsible for processing the content from other crews
        and creating a comprehensive summary. It analyzes the information,
        identifies key points, and structures the content in a clear,
        readable format suitable for the target audience.

        Returns
        -------
        Agent
            Configured agent for summarization with settings from the
            agents configuration file.

        Notes
        -----
        The agent configuration is loaded from the 'summarizer' section of
        the agents configuration file.
        """
        return Agent(config=self.agents_config["summarizer"], verbose=True)

    @task
    def summarization_task(self) -> Task:
        """
        Create the summarization task.

        This task defines the work to be performed by the summarizer agent.
        It specifies how the content should be processed, what format the
        summary should take, and where the output should be saved.

        Returns
        -------
        Task
            Configured task for summarization with settings from the
            tasks configuration file and output file specification.

        Notes
        -----
        Task configuration is loaded from the 'summarization_task' section of
        the tasks configuration file. The output is automatically saved to
        'output/report.md' for easy access.
        """
        return Task(
            config=self.tasks_config["summarization_task"],
            output_file="output/report.md",
        )

    @crew
    def summarization_crew(self) -> Crew:
        """
        Create the orchestrated summarization crew.

        This method assembles the summarizer agent and task into a working
        crew that processes content and produces comprehensive summaries.
        The crew uses sequential processing to ensure the summarization
        is completed before finishing.

        Returns
        -------
        Crew
            Configured crew with sequential processing, combining the
            summarizer agent with the summarization task.

        Notes
        -----
        The crew is designed to work as the final step in the QA flow,
        taking the output from other crews and producing a final,
        user-friendly summary report.
        """
        return Crew(
            agents=[self.summarizer()],
            tasks=[self.summarization_task()],
            process=Process.sequential,
            verbose=True,
        )
