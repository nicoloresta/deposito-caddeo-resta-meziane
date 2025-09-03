#!/usr/bin/env python
import json

from crewai import LLM
from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router


from rag_or_web_flow.crews.calculator_crew.calculator_crew import CalculatorCrew
from rag_or_web_flow.crews.rag_crew.rag_crew import RagCrew
from rag_or_web_flow.crews.web_search_crew.web_search_crew import WebSearchCrew

from rag_or_web_flow.tools.is_search_in_rag_output import IsSearchInRagOutput


class RAGOrWebState(BaseModel):
    search_info: IsSearchInRagOutput = None


class RAGOrWebFlow(Flow[RAGOrWebState]):
    @start()
    def start_conversation(self):
        user_input = input("Chat with LLM: ").strip()
        print(f"User input: {user_input}")

        llm = LLM(
            model="azure/gpt-4.1-nano",
            temperature=0,
            response_format=IsSearchInRagOutput,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a router that decides whether a user's query should be answered using a Retrieval-Augmented Generation (RAG) approach, a web search, or a calculator.",
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        result = llm.call(messages)

        json_result = json.loads(result)
        self.state.search_info = IsSearchInRagOutput(**json_result)

        print(self.state.search_info)

        return self.state.search_info.method

    @router(start_conversation)
    def route_tool(self, tool_name: str) -> str:
        return tool_name

    @listen("rag")
    def handle_rag(self):
        result = (
            RagCrew()
            .crew()
            .kickoff(inputs={"user_input": self.state.search_info.query})
        )

        print(result.raw)

    @listen("web")
    def handle_web_search(self):
        result = (
            WebSearchCrew()
            .crew()
            .kickoff(inputs={"search_query": self.state.search_info.query})
        )

        print("Summary:\n", result.raw)

    @listen("calculator")
    def handle_calculator(self):
        result = (
            CalculatorCrew()
            .crew()
            .kickoff(inputs={"user_input": self.state.search_info.query})
        )

        print("Calculator Result:\n", result.raw)


def kickoff():
    rag_or_web_flow = RAGOrWebFlow()
    rag_or_web_flow.kickoff()


def plot():
    rag_or_web_flow = RAGOrWebFlow()
    rag_or_web_flow.plot()


if __name__ == "__main__":
    kickoff()
