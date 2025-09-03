#!/usr/bin/env python
from pydantic import BaseModel

from crewai.flow import Flow, start

from crewai_deepeval.crews.calculator_crew.calculator_crew import CalculatorCrew

from deepeval.integrations.crewai import instrument_crewai

instrument_crewai()


class CalculatorFlowState(BaseModel):
    pass


class CalculatorFlow(Flow[CalculatorFlowState]):
    @start()
    def start_conversation(self):
        user_input = input("What calculation would you like to perform?\n>").strip()

        result = CalculatorCrew().crew().kickoff(inputs={"user_input": user_input})

        print("Calculator Result:\n", result.raw)


def kickoff():
    calculator_flow = CalculatorFlow()
    calculator_flow.kickoff()


def plot():
    calculator_flow = CalculatorFlow()
    calculator_flow.plot()


if __name__ == "__main__":
    kickoff()
