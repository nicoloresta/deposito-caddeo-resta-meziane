"""
Main module for the QA AI Agent system.

This module implements a flow-based system for managing user queries about topics,
with ethical checking, routing to appropriate crews (RAG or web search), and
summarization of results.

Classes
-------
GuideCreatorState
    Pydantic model for storing the current state of the guide creation flow.
GuideCreatorFlow
    Main flow class that orchestrates the entire QA process.

Functions
---------
kickoff
    Start the guide creator flow.
plot
    Generate a visualization of the flow.

Examples
--------
>>> from .main import kickoff
>>> kickoff()
"""

import json
import os
from re import search

from crewai import LLM
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel

from .crews.ethic_checker_crew.ethic_checker_crew import (EthicAnalysis,
                                                          EthicCheckerCrew)
from .crews.manager_choice_crew.manager_choice_crew import ManagerChoiceCrew
from .crews.rag_crew.rag_crew import RagCrew
from .crews.summarizer_crew.summarizer_crew import SummarizationCrew
from .crews.web_search_crew.web_search_crew import WebSearchCrew


class GuideCreatorState(BaseModel):
    """
    State model for the guide creator flow.

    Attributes
    ----------
    topic : str, default=""
        The current topic being processed by the flow.
    
    response : str, default=""
        The crew's response to be summarized.
    """

    topic: str = ""
    response: str = ""


class GuideCreatorFlow(Flow[GuideCreatorState]):
    """
    Flow to manage the user choice to search information about a topic or perform a calculation.

    This flow implements a complete pipeline for processing user queries:
    1. Gets user input for a topic
    2. Performs ethical analysis of the topic
    3. Routes to appropriate information retrieval method (RAG or web search)
    4. Summarizes the results

    The flow uses CrewAI's flow framework to orchestrate multiple AI agents
    working together to provide comprehensive answers to user queries.

    Attributes
    ----------
    state : GuideCreatorState
        The current state of the flow, containing the topic being processed.
    """

    @start("retry")
    def get_user_input(self):
        """
        Get the user choice for a topic to search.

        Prompts the user to input a topic they would like to search for
        and stores it in the flow state.

        Returns
        -------
        str
            The user's input topic.
        """
        self.state.topic = input("\nWhat topic would you like to search?\n>")
        return self.state.topic

    @router(get_user_input)
    def ethics_checker_router(self):
        """
        Check the ethical considerations for the guide topic.

        This method creates an ethics checking crew to analyze whether
        the user's requested topic is appropriate and ethical to process.

        Returns
        -------
        str
            Either "ethics_passed" if the topic is ethical, or "retry" if not.

        Notes
        -----
        If ethics analysis fails, the user is prompted to choose a different topic.
        """
        print("Checking ethical considerations...")

        ethic_checker_crew = EthicCheckerCrew()
        dict_analysis = ethic_checker_crew.crew().kickoff(
            inputs={"topic": self.state.topic}
        )
        analysis_data = json.loads(dict_analysis.raw)
        ethic_analysis_obj = EthicAnalysis(**analysis_data)

        if ethic_analysis_obj.is_ethical:
            print("Ethics analysis passed.")
            return "ethics_passed"
        else:
            print("Ethics analysis failed.")
            print(
                f"Ethics analysis details: {ethic_analysis_obj.reason}. Please choose a different request."
            )
            return "retry"

    @router("ethics_passed")
    def manager_choice_router(self):
        """
        Manage the user choice for information search or calculation.

        This method determines whether the user's topic is related to smartphones
        (which can use the RAG system) or requires general web search.

        Returns
        -------
        str
            Routing decision: "smartphone_rag", "general_search", or "retry".

        Notes
        -----
        Choice "1" routes to smartphone RAG, choice "2" routes to general search.
        Any other choice results in a retry.
        """
        print("Managing user choice...")

        clf_crew = ManagerChoiceCrew()
        choice = clf_crew.manager_choice_crew().kickoff(
            inputs={"topic": self.state.topic}
        )
        choice = choice.raw.strip()

        if choice == "1":
            print("The user wants to search for information about smartphones.")
            os.makedirs("output", exist_ok=True)
            return "smartphone_rag"
        if choice == "2":
            print(
                "The user wants to search for information about a topic not related to smartphones."
            )
            return "general_search"
        else:
            print("The user's choice did not match any known options. Retrying...")
            return "retry"

    @router("smartphone_rag")
    def run_smartphone_rag(self):
        """
        Run the smartphone RAG crew to gather information about smartphones.

        This method uses the RAG (Retrieval-Augmented Generation) system to
        search through pre-indexed smartphone documents and provide relevant
        information to the user's query.

        Returns
        -------
        str
            Either the RAG response content or routes to "general_search" if
            no relevant context is found.

        Notes
        -----
        If the RAG crew returns "0" (indicating no relevant context found),
        the request is automatically routed to the general search crew.
        """
        print("Running the smartphone RAG crew...")
        rag_crew = RagCrew()
        response = rag_crew.crew().kickoff(inputs={"topic": self.state.topic})
        if response.raw.strip() == "0":
            print("RAG crew did not find relevant context.")
            print("Trying to pass the request to the general search crew...")
            return "general_search"
        else:
            print("RAG crew completed and found relevant context.")
            self.state.response = response.raw.strip()
            return "summarize"

    @router("general_search")
    def run_general_search(self):
        """
        Run the general search crew to perform a web search.

        This method uses the web search crew to find information about topics
        that are not covered by the smartphone RAG system or when RAG fails
        to find relevant information.

        Returns
        -------
        str
            The raw response from the web search crew containing search results.
        """
        print("Running the general search crew...")
        search_crew = WebSearchCrew()
        response = search_crew.crew().kickoff(inputs={"prompt": self.state.topic})
        self.state.response = response.raw.strip()
        return "summarize"

    @listen("summarize")
    def summarization(self):
        """
        Summarize the response from the RAG or general search crew.

        This method takes the output from either the RAG crew or web search crew
        and creates a comprehensive summary of the information found. The summary
        is saved to an output file for the user to review.

        Parameters
        ----------
        to_summarize : str
            The content to be summarized, typically the output from RAG or web search.

        Notes
        -----
        The summarization crew processes the content and saves the result to
        'output/report.md'. This is the final step in the QA flow.
        """
        print("Summarizing the response from the RAG or general search crew...")
        crew = SummarizationCrew()
        response = crew.summarization_crew().kickoff(
            inputs={"prompt": self.state.response, "topic": self.state.topic}
        )
        print(response.raw)
        print("QA_AI_Agent task completed!")


def kickoff():
    """
    Start the guide creator flow.

    This function initializes and runs the complete GuideCreatorFlow,
    orchestrating the entire QA process from user input to final summary.

    Examples
    --------
    >>> kickoff()
    What topic would you like to search?
    > smartphones
    Checking ethical considerations...
    Ethics analysis passed.
    Managing user choice...
    ...
    """
    GuideCreatorFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/report.md to view it.")


def plot():
    """
    Generate a visualization of the flow.

    This function creates a visual representation of the GuideCreatorFlow
    structure, showing the relationships between different stages and
    decision points in the flow.

    Examples
    --------
    >>> plot()
    Flow visualization saved to qa_flow.html
    """
    flow = GuideCreatorFlow()
    flow.plot("qa_flow")
    print("Flow visualization saved to qa_flow.html")


if __name__ == "__main__":
    kickoff()
