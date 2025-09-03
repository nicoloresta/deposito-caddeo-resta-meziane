from crewai import Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from deepeval.integrations.crewai import Agent

from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from ...tools.calculator_tool import CalculatorTool


@CrewBase
class CalculatorCrew:
    """CalculatorCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def calculator(self) -> Agent:
        return Agent(
            config=self.agents_config["calculator"],  # type: ignore[index]
            verbose=True,
            tools=[CalculatorTool()],  # type: ignore[name]
        )

    @task
    def perform_calculation(self) -> Task:
        return Task(
            config=self.tasks_config["perform_calculation"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CalculatorCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
