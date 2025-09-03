from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

SupportedOperations = Literal["addition", "subtraction", "multiplication", "division"]


class CalculatorToolInput(BaseModel):
    """Input schema for CalculatorTool."""

    calculation: SupportedOperations = Field(
        ..., description="The type of calculation to perform."
    )

    a: float = Field(..., description="The first number.")
    b: float = Field(..., description="The second number.")


class CalculatorTool(BaseTool):
    name: str = "CalculatorTool"
    description: str = "Use this tool to perform various calculations."
    args_schema: Type[BaseModel] = CalculatorToolInput

    def _run(
        self,
        calculation: SupportedOperations,
        a: float,
        b: float,
    ) -> float:
        """
        Esegue un'operazione aritmetica elementare su due operandi.

        Parameters
        ----------
        calculation : {"addition", "subtraction", "multiplication", "division"}
            Tipo di operazione da eseguire.
        a : float
            Primo operando. Unità: adimensionale. Range: qualsiasi numero reale finito.
        b : float
            Secondo operando. Unità: adimensionale. Range: qualsiasi numero reale finito
            (per "division" deve essere diverso da 0).

        Returns
        -------
        float
            Esito dell'operazione. Unità: adimensionale.

        Raises
        ------
        ValueError
            Se `calculation` non è supportato.
        ValueError
            Se `calculation == "division"` e `b == 0`.

        Notes
        -----
        Complessità temporale: O(1).
        Complessità spaziale: O(1).

        Examples
        --------
        >>> from rag_or_web_flow.tools.calculator_tool import CalculatorTool
        >>> tool = CalculatorTool()
        >>> tool._run("addition", 2.5, 3.5)
        6.0
        >>> tool._run("subtraction", 10.0, 4.0)
        6.0
        >>> tool._run("multiplication", 3.0, 2.0)
        6.0
        >>> tool._run("division", 12.0, 2.0)
        6.0
        """

        match calculation:
            case "addition":
                return a + b
            case "subtraction":
                return a - b
            case "multiplication":
                return a * b
            case "division":
                if b == 0:
                    raise ValueError("Division by zero is not allowed.")
                return a / b
            case _:
                raise ValueError(f"Unsupported operation: {calculation}")
