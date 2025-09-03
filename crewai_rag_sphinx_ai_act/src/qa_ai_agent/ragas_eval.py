import os
import json
from pathlib import Path
from typing import Any, List

from crewai import Agent, Task, Crew
import yaml

from .crews.rag_crew.rag_crew import RagCrew
from .tools.qdrant_custom_tool import RagTool, RetrieverSettings

from ragas.evaluation import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas.dataset_schema import EvaluationDataset

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# def build_ragas_dataset(
#     questions: list[str],
#     ground_truth: list[str] = None,
# ) -> list[dict[str, Any]]:
#     rag_crew = RagCrew()

#     dataset = []
#     for i, q in enumerate(questions):
#         rewritten_query = rag_crew.rag_prompt_rewriter().kickoff(inputs={"topic": q})
#         rag_params = rag_crew.rag_define_search_params().kickoff(inputs={"topic":rewritten_query})

#         retriever_settings = RetrieverSettings(**json.loads(rag_params.raw))

#         retriever = RagTool()
#         context_str = retriever._run(query=rewritten_query, settings=retriever_settings)
        
#         chunks = context_str.split("\n\n")

#         final_response = rag_crew.rag_retriever().kickoff(
#             inputs={
#                 # "question": rewritten_query,
#                 # "context": context_str,
#                 'rag_define_search_params_task': rag_params,
#                 'rag_prompt_task': rewritten_query,
#             }
#         )

#         row = {
#             "user_input": q,
#             "retrieved_contexts": chunks,
#             "response": final_response,
#         }
#         if ground_truth:
#             row["reference"] = ground_truth[i]

#         dataset.append(row)
#     return dataset

def build_agent_from_yaml(
        config_path: Path, 
        agent_id: str, 
        output_model = None, 
        tools: List = None
    ):
    with open(config_path, 'r') as f:
        agents_config = yaml.safe_load(f)

    agent_config = agents_config[agent_id]
    agent = Agent(
        role=agent_config['role'],
        goal=agent_config['goal'],
        backstory=agent_config['backstory'],
        llm=agent_config['llm'],
        tools=tools,
        # allow_delegation=False,
        verbose=False,
        output_model=output_model
    )
    return agent

def build_task_from_yaml(
        agent, 
        config_path: Path, 
        task_id: str, 
        output_pydantic = None
    ):
    with open(config_path, 'r') as f:
        tasks_config = yaml.safe_load(f)

    tasks_config_config = tasks_config[task_id]
    task = Task(
        description=tasks_config_config['description'],
        expected_output=tasks_config_config['expected_output'],
        agent=agent,
        output_pydantic=output_pydantic
    )
    return task

def build_ragas_dataset(
    rag_tool: RagTool,
    questions: list[str],
    ground_truth: list[str] = None,
) -> list[dict[str, Any]]:
    
    # Manually instantiate agents and tasks to control tool usage
    # This assumes you have agent/task definitions in your RagCrew class
    # We will create them here to be explicit
    rag_crew_definitions = RagCrew()
    prompt_rewriter_agent = rag_crew_definitions.rag_prompt_rewriter()
    search_params_agent = build_agent_from_yaml(
        config_path=Path("./src/qa_ai_agent/crews/rag_crew/config/agents.yaml"),
        agent_id="rag_define_search_params",
        output_model=RetrieverSettings
    )
    rewrite_task = build_task_from_yaml(
        agent=prompt_rewriter_agent,
        config_path=Path("./src/qa_ai_agent/crews/rag_crew/config/tasks.yaml"),
        task_id="rag_prompt_task"
    )
    params_task = build_task_from_yaml(
        agent=search_params_agent,
        config_path=Path("./src/qa_ai_agent/crews/rag_crew/config/tasks.yaml"),
        task_id="rag_define_search_params_task",
        output_pydantic=RetrieverSettings
    )

    # Manually create the QA agent, passing the single RagTool instance
    rag_agent = build_agent_from_yaml(
        config_path=Path("./src/qa_ai_agent/crews/rag_crew/config/agents.yaml"), 
        tools=[rag_tool],
        agent_id="rag_retriever"
    )
    rag_task = build_task_from_yaml(
        agent=rag_agent,
        config_path=Path("./src/qa_ai_agent/crews/rag_crew/config/tasks.yaml"),
        task_id="rag_retrieval_task"
    )

    dataset = []
    for i, q in enumerate(questions):
        rewrite_crew = Crew(agents=[prompt_rewriter_agent], tasks=[rewrite_task], verbose=0)
        rewritten_query = rewrite_crew.kickoff(inputs={'topic': q})

        # Create and run a crew for the params task
        params_crew = Crew(agents=[search_params_agent], tasks=[params_task], verbose=0)
        rag_params_raw = params_crew.kickoff(
            inputs={'query': rewritten_query.raw}
        )
        print(f"rag_params_raw.pydantic = {rag_params_raw.pydantic}")
        retriever_settings = rag_params_raw.pydantic
        # retriever_settings = RetrieverSettings(**rag_params_raw.pydantic)

        # Use the single RagTool instance to get context
        context_str = rag_tool._run(query=rewritten_query.raw, settings=retriever_settings)
        chunks = context_str.split("\n\n")

        rag_crew = Crew(agents=[rag_agent], tasks=[rag_task], verbose=0)
        final_answer = rag_crew.kickoff(inputs={
            'rag_prompt_task': rewritten_query.raw,
            'rag_define_search_params_task': rag_params_raw.raw
        })

        row = {
            "user_input": q,
            "retrieved_contexts": chunks,
            "response": final_answer.raw,
        }
        if ground_truth:
            row["reference"] = ground_truth[i]

        dataset.append(row)
    return dataset

def main():
    # 5) Esempi di domande
    questions = [
        "Qual è la capacità della batteria del Galaxy S25 Ultra?",
        "Quale processore monta l’iPhone 16 Pro Max?",
        "Qual è il punto forte del Pixel 9 Pro XL secondo la recensione?",
        "Che differenza principale offre lo Xiaomi 14 Ultra?",
        "Quanto è grande il display dell’Honor Magic6 Pro?"
    ]

    # (opzionale) ground truth sintetica per correctness
    ground_truth = [
        "Il Galaxy S25 Ultra ha una batteria da 5000mAh con ricarica rapida a 45W e wireless a 25W.",
        "L’iPhone 16 Pro Max monta il chip A18 Pro.",
        "Il Pixel 9 Pro XL è ideale per chi cerca un’esperienza Android pura e offre ottime capacità fotografiche.",
        "Lo Xiaomi 14 Ultra si distingue per le sue capacità fotografiche avanzate e la qualità delle immagini.",
        "Honor Magic6 Pro ha un display AMOLED da 6.8 pollici con risoluzione 2800x1260."
    ]

    print("Building dataset ...")
    dataset = build_ragas_dataset(
        rag_tool=RagTool(),
        questions=questions,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    print(dataset)
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    print("Dataset built successfully.")

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    # Aggiungi correctness solo se tutte le righe hanno reference
    if all("reference" in row for row in dataset):
        metrics.append(answer_correctness)

    embedding_model = AzureOpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
    )

    # load Azure openai langchain llm
    llm = AzureChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
    )

    print("Evaluating with RAGAS...")
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=embedding_model,  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    # (facoltativo) salva per revisione umana
    df.to_csv("./output/ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")

if __name__ == "__main__":
    main()