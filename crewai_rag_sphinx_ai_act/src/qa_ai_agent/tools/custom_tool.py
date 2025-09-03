"""
Custom tools for the QA AI Agent system.

This module provides specialized tools for document processing, retrieval,
and RAG (Retrieval-Augmented Generation) operations. It includes document
loading, chunking, and a custom RAG tool that integrates with CrewAI.

Classes
-------
RetrieverSettings
    Configuration settings for document retrieval behavior.
ChunkingSettings
    Configuration settings for document chunking.
RagToolInput
    Input schema for the RAG tool.
RagTool
    Custom RAG tool for CrewAI integration.

Functions
---------
load_docs
    Load documents from a directory with support for multiple file types.
chunk_docs
    Split documents into smaller chunks for processing.

Examples
--------
>>> from .tools.custom_tool import RagTool, RetrieverSettings
>>> tool = RagTool(retriever_settings=RetrieverSettings(k=5))
>>> result = tool._run("What is the battery capacity of the Galaxy S22?")
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Type

from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (CSVLoader, DirectoryLoader,
                                                  PyPDFLoader, TextLoader)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()


@dataclass
class RetrieverSettings:
    """
    Settings for configuring the retriever behavior.

    This dataclass provides configuration options for the FAISS retriever,
    allowing customization of search strategy, result count, and diversity
    parameters.

    Parameters
    ----------
    search_type : str, default="similarity"
        Retrieval strategy. Supported values are ``"similarity"`` and ``"mmr"``
        (Maximal Marginal Relevance).
    k : int, default=4
        Number of top documents to return.
    fetch_k : int, default=10
        Number of candidate documents to fetch before ranking/filtering.
    mmr_lambda : float, default=0.5
        Trade-off between diversity and relevance when using ``search_type="mmr"``.
        Higher values favor relevance, lower values favor diversity.

    Examples
    --------
    >>> settings = RetrieverSettings(search_type="mmr", k=5, mmr_lambda=0.7)
    >>> settings.search_type
    'mmr'
    """

    search_type: str = "similarity"  # "mmr" o "similarity"
    k: int = 4
    fetch_k: int = 10
    mmr_lambda: float = 0.5


@dataclass
class ChunkingSettings:
    """
    Chunking configuration for splitting documents.

    This dataclass defines how documents should be split into smaller chunks
    for processing. The chunking process is crucial for effective retrieval
    and embedding generation.

    Parameters
    ----------
    chunk_size : int, default=1000
        Target character length for each chunk. Larger chunks provide more
        context but may be less focused.
    chunk_overlap : int, default=200
        Number of overlapping characters between adjacent chunks. Overlap
        helps maintain context continuity across chunk boundaries.

    Examples
    --------
    >>> settings = ChunkingSettings(chunk_size=1500, chunk_overlap=300)
    >>> settings.chunk_size
    1500
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200


def load_docs(
    path: Path, file_types: List[str] = [".txt", ".md", ".pdf", ".csv"]
) -> List[Document]:
    """
    Load documents from a directory.

    Supports ``.txt``, ``.md``, ``.pdf``, and ``.csv`` files using langchain
    loaders. Each file type is loaded with a suitable loader and aggregated
    into a single list of ``Document`` objects.

    Parameters
    ----------
    path : pathlib.Path
        Directory containing the files to load. Must exist and be accessible.
    file_types : list of str, optional
        File extensions to include. Defaults to ``[".txt", ".md", ".pdf", ".csv"]``.
        Unsupported file types are skipped with a warning message.

    Returns
    -------
    list of langchain.schema.Document
        Loaded documents with metadata preserved from the original files.

    Raises
    ------
    ValueError
        If ``path`` does not exist or is not accessible.

    Examples
    --------
    >>> from pathlib import Path
    >>> docs = load_docs(Path("./documents"), [".txt", ".pdf"])
    >>> len(docs)
    15
    >>> docs[0].metadata
    {'source': './documents/sample.txt'}
    """
    if not path.exists():
        raise ValueError("Invalid path provided")

    documents = []

    # Load different file types
    for file_type in file_types:
        try:
            if file_type == ".txt" or file_type == ".md":
                loader = DirectoryLoader(
                    path,
                    glob=f"**/*{file_type}",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                )
            elif file_type == ".pdf":
                loader = DirectoryLoader(
                    path, glob=f"**/*{file_type}", loader_cls=PyPDFLoader
                )
            elif file_type == ".csv":
                loader = DirectoryLoader(
                    path, glob=f"**/*{file_type}", loader_cls=CSVLoader
                )
            else:
                print(f"Unsupported file type: {file_type}")
                continue

            docs = loader.load()
            documents.extend(docs)
            print(f"Loaded {len(docs)} {file_type} documents")

        except Exception as e:
            print(f"Error loading {file_type} files: {str(e)}")

    return documents


def chunk_docs(docs: List[Document], settings: ChunkingSettings):
    """
    Split documents into smaller chunks.

    Uses ``RecursiveCharacterTextSplitter`` with the provided settings to
    generate overlapping chunks suitable for embedding and retrieval.
    The splitter attempts to break on natural boundaries like paragraphs,
    sentences, and punctuation.

    Parameters
    ----------
    docs : list of langchain.schema.Document
        Documents to split. Each document should have a ``page_content``
        attribute containing the text to be chunked.
    settings : ChunkingSettings
        Chunking configuration specifying chunk size and overlap.

    Returns
    -------
    list of langchain.schema.Document
        Chunked documents with preserved metadata. Each chunk maintains
        the original document's metadata while containing a subset of
        the content.

    Examples
    --------
    >>> from .tools.custom_tool import ChunkingSettings
    >>> settings = ChunkingSettings(chunk_size=500, chunk_overlap=100)
    >>> chunks = chunk_docs(documents, settings)
    >>> len(chunks)
    25
    >>> chunks[0].page_content
    'First chunk content...'
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", "", "---"],
    )

    return text_splitter.split_documents(docs)


class RagToolInput(BaseModel):
    """
    Input schema for ``RagTool``.

    This Pydantic model defines the expected input format for the RAG tool,
    ensuring that queries are properly validated before processing.

    Parameters
    ----------
    query : str
        Query for retrieving the most relevant documents. Should be a
        natural language question or search term.

    Examples
    --------
    >>> input_data = RagToolInput(query="What is the battery capacity?")
    >>> input_data.query
    'What is the battery capacity?'
    """

    query: str = Field(..., description="Query for retrieving most relevant documents")


class RagTool(BaseTool):
    """
    Retrieval-Augmented Generation (RAG) tool.

    The tool retrieves relevant document chunks from a FAISS index and can be
    used within CrewAI workflows to augment generation with external context.
    It automatically handles document loading, chunking, and indexing if no
    existing index is found.

    The tool supports both similarity search and Maximal Marginal Relevance (MMR)
    retrieval strategies, allowing for customization of result relevance vs.
    diversity trade-offs.

    Attributes
    ----------
    name : str
        Tool identifier used by CrewAI.
    description : str
        Human-readable description of the tool's functionality.
    args_schema : Type[BaseModel]
        Input validation schema (RagToolInput).
    _retriever : Any
        Private attribute storing the configured FAISS retriever.

    Examples
    --------
    >>> tool = RagTool(
    ...     rag_path=Path("./my_rag_index"),
    ...     docs_path=Path("./my_documents"),
    ...     retriever_settings=RetrieverSettings(k=5)
    ... )
    >>> result = tool._run("What features does the device have?")
    """

    name: str = "Rag Tool"
    description: str = (
        "A tool for retrieving and generating information using RAG (Retrieval-Augmented Generation) techniques."
    )
    args_schema: Type[BaseModel] = RagToolInput

    # Define retriever as a private attribute
    _retriever: Any = None

    def __init__(
        self,
        embedding_model: Any = None,
        rag_path: Path = Path("./rsc/rag"),
        docs_path: Path = Path("./rsc/docs"),
        retriever_settings: RetrieverSettings = RetrieverSettings(),
        chunk_settings: ChunkingSettings = ChunkingSettings(),
    ):
        """
        Initialize the RAG tool.

        Builds or loads a FAISS vector store from ``rag_path``. If a previously
        saved index exists in ``rag_path``, it is loaded and a retriever is
        configured. Otherwise, documents are loaded, chunked, and a new vector
        store is created and saved.

        Parameters
        ----------
        embedding_model : Any, optional
            LangChain embedding model instance. If None, creates an Azure OpenAI
            embeddings model using environment variables.
        rag_path : pathlib.Path, default=Path("./rsc/rag")
            Directory containing source documents or an existing FAISS index.
            If the directory doesn't exist, it will be created.
        docs_path : pathlib.Path, default=Path("./rsc/docs")
            Directory containing source documents for indexing. Only used when
            creating a new index.
        retriever_settings : RetrieverSettings, optional
            Retrieval configuration used when creating the retriever.
            Defaults to RetrieverSettings().
        chunk_settings : ChunkingSettings, optional
            Chunking configuration for splitting documents before indexing.
            Defaults to ChunkingSettings().

        Notes
        -----
        The tool automatically detects existing FAISS indices (``index.faiss``
        and ``index.pkl`` files) and loads them if available. Otherwise, it
        processes documents from ``docs_path`` to create a new index.

        Environment variables required for Azure OpenAI embeddings:
        - EMBEDDING_MODEL: Model name
        - AZURE_API_BASE: API endpoint
        - AZURE_API_KEY: API key
        - AZURE_API_VERSION: API version

        Examples
        --------
        >>> tool = RagTool(
        ...     rag_path=Path("./custom_rag"),
        ...     retriever_settings=RetrieverSettings(search_type="mmr", k=3)
        ... )
        """
        # Call parent constructor first
        super().__init__()

        # Initialize embedding model if not provided
        if embedding_model is None:
            embedding_model = AzureOpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL"),
                azure_endpoint=os.getenv("AZURE_API_BASE"),
                api_key=os.getenv("AZURE_API_KEY"),
                openai_api_version=os.getenv("AZURE_API_VERSION"),
            )
        # checks if in path there is already the rag
        if (rag_path / "index.faiss").exists() and (rag_path / "index.pkl").exists():
            print("RAG index already exists.")
            vector_store = FAISS.load_local(
                rag_path, embedding_model, allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(
                search_type=retriever_settings.search_type,
                search_kwargs={
                    "k": retriever_settings.k,
                    "fetch_k": retriever_settings.fetch_k,
                    "lambda_mult": retriever_settings.mmr_lambda,
                },
            )
            # Store retriever as a private attribute
            self._retriever = retriever
        else:
            docs = load_docs(docs_path)
            chunks = chunk_docs(docs, chunk_settings)
            vector_store = FAISS.from_documents(
                documents=chunks, embedding=embedding_model
            )
            # save vector store into path
            vector_store.save_local(folder_path=rag_path)
            retriever = vector_store.as_retriever(
                search_type=retriever_settings.search_type,
                search_kwargs={
                    "k": retriever_settings.k,
                    "fetch_k": retriever_settings.fetch_k,
                    "lambda_mult": retriever_settings.mmr_lambda,
                },
            )
            # Store retriever as a private attribute
            self._retriever = retriever

    def _run(self, query: str) -> str:
        """
        Run a retrieval query and return concatenated contents.

        This method is called by CrewAI when the tool is invoked. It uses
        the configured FAISS retriever to find the most relevant document
        chunks for the given query and returns them in a formatted string.

        Parameters
        ----------
        query : str
            Natural language query used to retrieve relevant chunks.
            Should be a clear, specific question or search term.

        Returns
        -------
        str
            Concatenation of the retrieved documents' ``page_content``,
            separated by blank lines. Each chunk is prefixed with its
            source information for traceability.

        Examples
        --------
        >>> tool = RagTool()
        >>> result = tool._run("What is the screen resolution?")
        >>> print(result[:100])
        [Source: ./rsc/docs/specs.txt]
        The device features a 6.1-inch display with...
        """
        best_chunks = self._retriever.invoke(query)
        # Process the best_chunks to generate a response including the source
        return "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in best_chunks
            ]
        )
