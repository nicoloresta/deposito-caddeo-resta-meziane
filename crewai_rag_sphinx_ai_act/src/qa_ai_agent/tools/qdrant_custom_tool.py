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
import re
from typing import Any, List, Tuple, Type

from crewai import LLM
import numpy as np
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (CSVLoader, DirectoryLoader,
                                                  PyPDFLoader, TextLoader)
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, FieldCondition, Filter,
                                       HnswConfigDiff, MatchText,
                                       OptimizersConfigDiff, PayloadSchemaType,
                                       PointStruct, ScalarQuantization,
                                       ScalarQuantizationConfig, SearchParams,
                                       VectorParams)

load_dotenv()

@dataclass
class QdrantSettings:
    qdrant_url: str = "localhost:6333"
    collection: str = "rag_chunks"  

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

class RetrieverSettings(BaseModel):
    # =========================
    # Hybrid Search Configuration
    # =========================
    top_n_semantic: int = 30
    """
        Number of top semantic search candidates to retrieve initially.
        
        Semantic Search Candidates:
        - Low values (10-20): Fast retrieval, may miss relevant results
        - Medium values (30-50): Good balance between speed and recall
        - High values (100+): Maximum recall, slower performance
        
        Performance Impact:
        - Retrieval time: Linear increase with candidate count
        - Memory usage: Linear increase with candidate count
        - Quality: Diminishing returns beyond 50-100 candidates
        
        Tuning Guidelines:
        - Small collections (<1000 docs): 20-30 candidates
        - Medium collections (1000-10000 docs): 30-50 candidates
        - Large collections (10000+ docs): 50-100 candidates
    """
    
    top_n_text: int = 100
    """
        Maximum number of text-based matches to consider for hybrid fusion.
        
        Text Search Scope:
        - Low values (50): Fast text filtering, may miss relevant matches
        - Medium values (100): Good balance between speed and coverage
        - High values (200+): Maximum text coverage, slower performance
        
        Hybrid Search Strategy:
        - Text search acts as a pre-filter for semantic results
        - Higher values improve the quality of text-semantic fusion
        - Optimal value depends on collection size and query complexity
    """
    
    final_k: int = 6
    """
        Final number of results to return after all processing steps.
        
        Result Count Considerations:
        - User experience: 3-5 results for simple queries, 5-10 for complex ones
        - Context window: Align with LLM context limits (e.g., 6-8 chunks for GPT-3.5)
        - Diversity: Higher values allow MMR to select more diverse results
        
        LLM Integration:
        - GPT-3.5: 6-8 chunks typically fit in context
        - GPT-4: 8-12 chunks can be processed
        - Claude: 6-10 chunks work well
    """
    
    alpha: float = 0.75
    """
        Weight for semantic similarity in hybrid score fusion (0.0 to 1.0).
        
        Alpha Parameter Behavior:
        - alpha = 0.0: Pure text-based ranking (BM25, keyword matching)
        - alpha = 0.5: Equal weight for semantic and text relevance
        - alpha = 0.75: Semantic similarity prioritized (current setting)
        - alpha = 1.0: Pure semantic ranking (cosine similarity only)
        
        Use Case Recommendations:
        - Technical queries: 0.7-0.9 (semantic understanding important)
        - Factual queries: 0.5-0.7 (balanced approach)
        - Keyword searches: 0.3-0.5 (text matching more important)
        - Conversational queries: 0.6-0.8 (semantic context matters)
        
        Tuning Strategy:
        - Start with 0.75 for general use
        - Increase if semantic results seem irrelevant
        - Decrease if text matching is too weak
    """
    
    text_boost: float = 0.20
    """
        Additional score boost for results that match both semantic and text criteria.
        
        Text Boost Mechanism:
        - Applied additively to fused scores
        - Encourages results that satisfy both search strategies
        - Helps surface highly relevant content that matches multiple criteria
        
        Boost Value Guidelines:
        - Low boost (0.1-0.2): Subtle preference for hybrid matches
        - Medium boost (0.2-0.4): Strong preference for hybrid matches
        - High boost (0.5+): Heavy preference, may dominate ranking
        
        Optimal Settings:
        - General use: 0.15-0.25
        - Technical content: 0.20-0.30
        - Factual queries: 0.10-0.20
    """
    
    # =========================
    # MMR (Maximal Marginal Relevance) Configuration
    # =========================
    use_mmr: bool = True
    """
        Whether to use MMR for result diversification and redundancy reduction.
        
        MMR Benefits:
        - Reduces redundant results with similar content
        - Improves coverage of different aspects of the query
        - Better user experience with diverse information
        
        MMR Trade-offs:
        - Slightly slower than simple top-K selection
        - May reduce absolute relevance scores
        - Better for exploratory queries, worse for specific fact retrieval
        
        Alternatives:
        - False: Simple top-K selection (faster, may have redundancy)
        - True: MMR diversification (slower, better diversity)
    """
    
    mmr_lambda: float = 0.6
    """
        MMR diversification parameter balancing relevance vs. diversity (0.0 to 1.0).
        
        Lambda Parameter Behavior:
        - lambda = 0.0: Pure diversity (ignore relevance, maximize difference)
        - lambda = 0.5: Balanced relevance and diversity
        - lambda = 0.6: Slight preference for relevance (current setting)
        - lambda = 1.0: Pure relevance (ignore diversity, top-K selection)
        
        Use Case Recommendations:
        - Research queries: 0.4-0.6 (diverse perspectives important)
        - Factual queries: 0.7-0.9 (relevance more important)
        - Exploratory queries: 0.3-0.5 (diversity valuable)
        - Specific searches: 0.8-1.0 (precision over diversity)
        
        Tuning Guidelines:
        - Start with 0.6 for general use
        - Decrease if results seem too similar
        - Increase if results seem too diverse
    """

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
            for doc in docs:
                doc.metadata["file_type"] = file_type
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
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=settings.chunk_size,
    #     chunk_overlap=settings.chunk_overlap,
    #     separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", "", "---"],
    # )

    chunks = []

    md_regex = re.compile(r'(?P<section>#+\s+.+)\n\n*(?P<content>[^#]+)')

    for doc in docs:
        if doc.metadata.get("file_type") == ".md":
            matches = md_regex.finditer(doc.page_content)
            for match in matches:
                section = match.group("section").strip()
                content = match.group("content").strip()
                # add the matched chunk into chunks
                chunks.append(Document(
                    page_content=content, 
                    metadata={
                        "section": section,
                    }
                ))

    return chunks

class EmbeddingModel:

    def get_sentence_embedding_dimension(self) -> int:
        raise NotImplementedError("get_sentence_embedding_dimension method not implemented!")

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("embed_documents method not implemented!")
    
    def embed_query(self, query: str) -> List[float]:
        raise NotImplementedError("embed_query method not implemented!")

class AzureOpenAIEmbeddings(EmbeddingModel):

    def __init__(self, model: str, azure_endpoint: str, api_key: str, openai_api_version: str):
        from langchain_openai import AzureOpenAIEmbeddings as LangchainAzureOpenAIEmbeddings
        self._client = LangchainAzureOpenAIEmbeddings(
            model=model,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            openai_api_version=openai_api_version,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return 1536

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self._client.embed_documents(docs)

    def embed_query(self, query: str) -> List[float]:
        return self._client.embed_query(query)

class HFEmbeddings(EmbeddingModel):

    def __init__(self, model_name: str):
        from langchain_community.embeddings import HuggingFaceEmbeddings as LangchainHFEmbeddings
        self._client = LangchainHFEmbeddings(model_name=model_name)

    def get_sentence_embedding_dimension(self) -> int:
        return self._client.get_sentence_embedding_dimension()

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self._client.embed_documents(docs)
    
    def embed_query(self, query: str) -> List[float]:
        return self._client.embed_query(query)

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
    settings: RetrieverSettings = Field(
        default=RetrieverSettings(),
        description="Settings for document retrieval behavior",
    )

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
    client: QdrantClient = None
    embedding_model: EmbeddingModel = None
    collection_name: str = None

    def __recreate_collection_for_rag(self, collection_name: str, vector_size: int) -> None:
        """
            Create or recreate a Qdrant collection optimized for RAG (Retrieval-Augmented Generation).
            
            This function sets up a vector database collection with optimal configuration for
            semantic search, including HNSW indexing, payload indexing, and quantization.
            
            Args:
                client: Qdrant client instance for database operations
                settings: Configuration object containing collection parameters
                vector_size: Dimension of the embedding vectors (e.g., 384 for MiniLM-L6)
                
            Collection Architecture:
            - Vector storage: Dense vectors for semantic similarity search
            - Payload storage: Metadata and text content for retrieval
            - Indexing: HNSW for approximate nearest neighbor search
            - Quantization: Scalar quantization for memory optimization
                
            Distance Metric Selection:
            - Cosine distance: Normalized similarity, good for semantic embeddings
            - Alternatives: Euclidean (L2), Manhattan (L1), Dot product
            - Cosine preferred for normalized embeddings (sentence-transformers)
                
            HNSW Index Configuration:
            - m=32: Average connections per node (higher = better quality, more memory)
            - ef_construct=256: Search depth during construction (higher = better quality, slower build)
            - Trade-offs: Higher values improve recall but increase memory and build time
                
            Optimizer Configuration:
            - default_segment_number=2: Parallel processing segments
            - Benefits: Faster indexing, better resource utilization
            - Considerations: More segments = more memory overhead
                
            Quantization Strategy:
            - Scalar quantization: Reduces vector precision from float32 to int8
            - Memory savings: ~4x reduction in vector storage
            - Quality impact: Minimal impact on search accuracy
            - always_ram=False: Vectors stored on disk, loaded to RAM as needed
                
            Payload Indexing Strategy:
            - Text index: Full-text search capabilities (BM25 scoring)
            - Keyword indices: Fast exact matching and filtering
            - Performance: Significantly faster than unindexed field searches
                
            Collection Lifecycle:
            - recreate_collection: Drops existing collection and creates new one
            - Use case: Development/testing, major schema changes
            - Production: Consider using create_collection + update_collection_info
                
            Performance Considerations:
            - Build time: HNSW construction scales with collection size
            - Memory usage: Vectors loaded to RAM during search
            - Storage: Quantized vectors + payload data
            - Query latency: HNSW provides sub-millisecond search times
                
            Scaling Guidelines:
            - Small collections (<100K vectors): Current settings optimal
            - Medium collections (100K-1M vectors): Increase m to 48-64
            - Large collections (1M+ vectors): Consider multiple collections or sharding
        """
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=32,             # grado medio del grafo HNSW (maggiore = più memoria/qualità)
                ef_construct=256  # ampiezza lista candidati in fase costruzione (qualità/tempo build)
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2  # parallelismo/segmentazione iniziale
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type="int8", always_ram=False)  # on-disk quantization dei vettori
            ),
        )

        # Indice full-text sul campo 'text' per filtri MatchText
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT
        )

        # Indici keyword per filtri esatti / velocità nei filtri
        for key in ["doc_id", "source", "title", "lang"]:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=key,
                field_schema=PayloadSchemaType.KEYWORD
            )

    def __build_points(self, chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
        pts: List[PointStruct] = []
        for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
            payload = {
                "doc_id": doc.metadata.get("id"),
                "source": doc.metadata.get("source"),
                "title": doc.metadata.get("title"),
                "lang": doc.metadata.get("lang", "en"),
                "text": doc.page_content,
                "chunk_id": i - 1
            }
            pts.append(PointStruct(id=i, vector=vec, payload=payload))
        return pts

    def __qdrant_text_prefilter_ids(
        self,
        query: str,
        max_hits: int
    ) -> List[int]:
        """
        Usa l'indice full-text su 'text' per prefiltrare i punti che contengono parole chiave.
        Non restituisce uno score BM25: otteniamo un sottoinsieme di id da usare come boost.
        """
        # Scroll con filtro MatchText per ottenere id dei match testuali
        # (nota: scroll è paginato; qui prendiamo solo i primi max_hits per semplicità)
        matched_ids: List[int] = []
        next_page = None
        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="text", match=MatchText(text=query))]
                ),
                limit=min(256, max_hits - len(matched_ids)),
                offset=next_page,
                with_payload=False,
                with_vectors=False,
            )
            matched_ids.extend([p.id for p in points])
            if not next_page or len(matched_ids) >= max_hits:
                break
        return matched_ids

    def __mmr_select(
        self,
        query_vec: List[float],
        candidates_vecs: List[List[float]],
        k: int,
        lambda_mult: float
    ) -> List[int]:
        """
            Select diverse results using Maximal Marginal Relevance (MMR) algorithm.
            
            MMR balances relevance to the query with diversity among selected results,
            reducing redundancy and improving information coverage. This is particularly
            useful for RAG systems where diverse context provides better generation.
            
            Args:
                query_vec: Query embedding vector for relevance calculation
                candidates_vecs: List of candidate document embedding vectors
                k: Number of results to select
                lambda_mult: MMR parameter balancing relevance vs. diversity (0.0 to 1.0)
                
            Returns:
                List[int]: Indices of selected candidates in order of selection
                
            MMR Algorithm Overview:
            
            The algorithm iteratively selects candidates that maximize the MMR score:
            
            MMR_score(i) = λ × Relevance(i, query) - (1-λ) × max_similarity(i, selected)
            
            Where:
            - λ (lambda_mult): Weight for relevance vs. diversity
            - Relevance(i, query): Cosine similarity between candidate i and query
            - max_similarity(i, selected): Maximum similarity between candidate i and already selected items
                
            Algorithm Steps:
            
            1. INITIALIZATION:
            - Calculate relevance scores for all candidates vs. query
            - Select the highest-scoring candidate as the first result
            - Initialize selected and remaining candidate sets
                
            2. ITERATIVE SELECTION:
            - For each remaining position, calculate MMR score for all candidates
            - MMR score balances query relevance with diversity from selected items
            - Select candidate with highest MMR score
            - Update selected and remaining sets
                
            3. TERMINATION:
            - Continue until k candidates selected or no more candidates available
            - Return indices in selection order
                
            Mathematical Foundation:
            
            Cosine Similarity:
            - cos(a,b) = (a·b) / (||a|| × ||b||)
            - Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
            - Normalized vectors typically have values in [0, 1] range
                
            MMR Score Calculation:
            - Relevance term: λ × cos(query, candidate)
            - Diversity term: (1-λ) × max(cos(candidate, selected_i))
            - Higher relevance increases score, higher similarity to selected decreases score
                
            Lambda Parameter Behavior:
            
            λ = 0.0 (Pure Diversity):
            - Only diversity matters, relevance ignored
            - Results may be irrelevant to query
            - Useful for exploratory search
                
            λ = 0.5 (Balanced):
            - Equal weight for relevance and diversity
            - Good compromise for general use
            - Moderate redundancy reduction
                
            λ = 0.6 (Current Setting):
            - Slight preference for relevance
            - Good diversity while maintaining relevance
            - Recommended for most RAG applications
                
            λ = 1.0 (Pure Relevance):
            - Only relevance matters, diversity ignored
            - Equivalent to simple top-K selection
            - May have redundant results
                
            Performance Characteristics:
            
            Time Complexity:
            - O(k × n) where k = results to select, n = total candidates
            - Each iteration processes all remaining candidates
            - Quadratic complexity in worst case (k ≈ n)
                
            Space Complexity:
            - O(n) for storing vectors and similarity scores
            - O(k) for selected indices
            - O(n) for remaining candidate set
                
            Memory Usage:
            - Vector storage: All candidate vectors loaded in memory
            - Similarity cache: Relevance scores computed once
            - Selection state: Small overhead for tracking
                
            Quality Metrics:
            
            Relevance Preservation:
            - Higher lambda values preserve more relevance
            - Lower lambda values may sacrifice relevance for diversity
            - Optimal balance depends on use case
                
            Diversity Improvement:
            - MMR significantly reduces redundancy compared to top-K
            - Diversity increases as lambda decreases
            - Measurable improvement in information coverage
                
            User Experience:
            - Less repetitive results
            - Better coverage of different aspects
            - More informative context for LLM generation
                
            Use Case Recommendations:
            
            Research & Exploration:
            - λ = 0.3-0.5: Maximize diversity for comprehensive understanding
            - Higher k values: More diverse perspectives
                
            Factual Queries:
            - λ = 0.7-0.9: Prioritize relevance for accurate information
            - Lower k values: Focus on most relevant results
                
            Technical Documentation:
            - λ = 0.5-0.7: Balance relevance with diverse technical perspectives
            - Moderate k values: Comprehensive technical coverage
                
            Conversational AI:
            - λ = 0.6-0.8: Good relevance with some diversity
            - Higher k values: Rich context for generation
                
            Tuning Guidelines:
            
            For Maximum Diversity:
            - Decrease lambda to 0.3-0.5
            - Increase k to 8-12 results
            - Monitor relevance quality
                
            For Maximum Relevance:
            - Increase lambda to 0.8-1.0
            - Decrease k to 3-6 results
            - Accept some redundancy
                
            For Balanced Results:
            - Use lambda = 0.6-0.7 (current setting)
            - Moderate k values (6-8)
            - Good compromise for most applications
                
            Implementation Notes:
            
            Numerical Stability:
            - Small epsilon (1e-12) added to prevent division by zero
            - Cosine similarity handles normalized vectors robustly
            - Float precision sufficient for similarity calculations
                
            Edge Cases:
            - Empty candidate list: Returns empty result
            - k > candidates: Returns all candidates
            - Single candidate: Returns that candidate regardless of lambda
                
            Optimization Opportunities:
            - Vector similarity could be pre-computed and cached
            - Parallel processing for large candidate sets
            - Early termination for very low diversity scores
        """
        import numpy as np
        V = np.array(candidates_vecs, dtype=float)
        q = np.array(query_vec, dtype=float)

        def cos(a, b):
            na = (a @ a) ** 0.5 + 1e-12
            nb = (b @ b) ** 0.5 + 1e-12
            return float((a @ b) / (na * nb))

        sims = [cos(v, q) for v in V]
        selected: List[int] = []
        remaining = set(range(len(V)))

        while len(selected) < min(k, len(V)):
            if not selected:
                # pick the highest similarity first
                best = max(remaining, key=lambda i: sims[i])
                selected.append(best)
                remaining.remove(best)
                continue
            best_idx = None
            best_score = -1e9
            for i in remaining:
                max_div = max([cos(V[i], V[j]) for j in selected]) if selected else 0.0
                score = lambda_mult * sims[i] - (1 - lambda_mult) * max_div
                if score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    def __hybrid_search(
        self,
        settings: RetrieverSettings,
        query: str,
    ):
        """
        Perform hybrid search combining semantic similarity and text-based matching.
        
        This function implements a sophisticated retrieval strategy that leverages both
        semantic understanding and traditional text search to provide high-quality,
        relevant results with minimal redundancy.
        
        Args:
            client: Qdrant client for database operations
            settings: Configuration object containing search parameters
            query: User's search query string
            embeddings: Embedding model for semantic search
            
        Returns:
            List[ScoredPoint]: Ranked list of relevant document chunks
            
        Hybrid Search Strategy Overview:
        
        1. SEMANTIC SEARCH (Vector Similarity):
        - Converts query to embedding vector
        - Performs approximate nearest neighbor search using HNSW index
        - Retrieves top_n_semantic candidates based on cosine similarity
        - Provides semantic understanding of query intent
            
        2. TEXT-BASED PREFILTERING:
        - Uses full-text search capabilities (BM25 scoring)
        - Identifies documents containing query keywords/phrases
        - Creates a set of text-relevant document IDs
        - Acts as a relevance filter for semantic results
            
        3. SCORE FUSION & NORMALIZATION:
        - Normalizes semantic scores to [0,1] range for fair comparison
        - Applies alpha weight to balance semantic vs. text relevance
        - Adds text_boost for results matching both criteria
        - Creates unified relevance scoring
            
        4. RESULT DIVERSIFICATION (Optional MMR):
        - Applies Maximal Marginal Relevance to reduce redundancy
        - Balances relevance with diversity using mmr_lambda parameter
        - Selects final_k results from top candidates
            
        Algorithm Flow:
        
        Phase 1: Semantic Retrieval
        - Query embedding generation
        - HNSW-based vector search
        - Score normalization for fusion
            
        Phase 2: Text Matching
        - Full-text search with MatchText filter
        - ID collection for hybrid scoring
        - Performance optimization with pagination
            
        Phase 3: Score Fusion
        - Linear combination of semantic and text scores
        - Boost application for hybrid matches
        - Ranking by fused scores
            
        Phase 4: Result Selection
        - Top-N selection or MMR diversification
        - Final result ordering and return
            
        Performance Characteristics:
        
        Time Complexity:
        - Semantic search: O(log n) with HNSW index
        - Text search: O(m) where m is text matches
        - Score fusion: O(k) where k is semantic candidates
        - MMR: O(k²) for diversity computation
            
        Memory Usage:
        - Vector storage: Quantized vectors in memory
        - Score storage: Temporary arrays for fusion
        - Result storage: Final selected points
            
        Quality Metrics:
        
        Recall (Completeness):
        - Semantic search: High recall for conceptual queries
        - Text search: High recall for keyword queries
        - Hybrid approach: Combines strengths of both
            
        Precision (Relevance):
        - Score fusion: Balances multiple relevance signals
        - Text boost: Rewards multi-criteria matches
        - MMR: Reduces redundant results
            
        Diversity:
        - MMR algorithm: Maximizes information coverage
        - Lambda parameter: Controls diversity vs. relevance trade-off
        - Result variety: Better user experience
            
        Tuning Guidelines:
        
        For High Precision:
        - Increase alpha (0.8-0.9): Prioritize semantic similarity
        - Increase text_boost (0.3-0.5): Reward text matches
        - Decrease mmr_lambda (0.7-0.9): Prioritize relevance
            
        For High Recall:
        - Increase top_n_semantic (50-100): More candidates
        - Increase top_n_text (150-200): More text matches
        - Decrease alpha (0.5-0.7): Balance search strategies
            
        For High Diversity:
        - Enable MMR (use_mmr=True)
        - Decrease mmr_lambda (0.3-0.6): Prioritize diversity
        - Increase final_k (8-12): More diverse results
            
        Use Case Optimizations:
        
        Technical Documentation:
        - High alpha (0.8-0.9): Semantic understanding critical
        - High text_boost (0.3-0.4): Technical terms important
        - MMR enabled: Diverse technical perspectives
            
        General Knowledge:
        - Balanced alpha (0.6-0.8): Both strategies valuable
        - Moderate text_boost (0.2-0.3): Balanced approach
        - MMR enabled: Comprehensive coverage
            
        Factual Queries:
        - High alpha (0.7-0.9): Semantic context important
        - Low text_boost (0.1-0.2): Facts over style
        - MMR optional: Precision over diversity
        """
        # (1) semantica
        qv = self.embedding_model.embed_query(query)
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=qv,
            limit=settings.top_n_semantic,
            with_payload=True,
            with_vectors=True,
            search_params=SearchParams(
                hnsw_ef=256,  # ampiezza lista in fase di ricerca (recall/latency)
                exact=False   # True = ricerca esatta (lenta); False = ANN HNSW
            ),
        )
        sem = res.points
        if not sem:
            return []

        # (2) full-text prefilter (id)
        text_ids = set(self.__qdrant_text_prefilter_ids(query, settings.top_n_text))

        # Normalizzazione score semantici per fusione
        scores = [p.score for p in sem]
        smin, smax = min(scores), max(scores)
        def norm(x):  # robusto al caso smin==smax
            return 1.0 if smax == smin else (x - smin) / (smax - smin)

        # (3) fusione con boost testuale
        fused: List[Tuple[int, float, Any]] = []  # (idx, fused_score, point)
        for idx, p in enumerate(sem):
            base = norm(p.score)                    # [0..1]
            fuse = settings.alpha * base
            if p.id in text_ids:
                fuse += settings.text_boost         # boost additivo
            fused.append((idx, fuse, p))

        # ordina per fused_score desc
        fused.sort(key=lambda t: t[1], reverse=True)

        # MMR opzionale per diversificare i top-K
        if settings.use_mmr:
            qv = self.embedding_model.embed_query(query)
            # prendiamo i primi N dopo fusione (es. 30) e poi MMR per final_k
            N = min(len(fused), max(settings.final_k * 5, settings.final_k))
            cut = fused[:N]
            vecs = [sem[i].vector for i, _, _ in cut]
            mmr_idx = self.__mmr_select(qv, vecs, settings.final_k, settings.mmr_lambda)
            picked = [cut[i][2] for i in mmr_idx]
            return picked

        # altrimenti, prendi i primi final_k dopo fusione
        return [p for _, _, p in fused[:settings.final_k]]

    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        # rag_path: Path = Path("./rsc/rag"),
        docs_path: Path = Path("./rsc/docs"),
        qdrant_settings: QdrantSettings = QdrantSettings(),
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
        self.client = QdrantClient(url=qdrant_settings.qdrant_url)
        self.collection_name = qdrant_settings.collection

        # Initialize embedding model if not provided
        if embedding_model is None:
            # raise ValueError("embedding_model must be provided!")
            self.embedding_model = AzureOpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL"),
                azure_endpoint=os.getenv("AZURE_API_BASE"),
                api_key=os.getenv("AZURE_API_KEY"),
                openai_api_version=os.getenv("AZURE_API_VERSION"),
            )
        else:
            self.embedding_model = embedding_model
        
        docs = load_docs(docs_path)
        chunks = chunk_docs(docs, chunk_settings)
        print(f"Chunks:\n{chunks}")
        input('> ')
        vector_size = self.embedding_model.get_sentence_embedding_dimension()
        self.__recreate_collection_for_rag(qdrant_settings.collection, vector_size)
        vecs = self.embedding_model.embed_documents([c.page_content for c in chunks])
        points = self.__build_points(chunks, vecs)
        self.client.upsert(collection_name=qdrant_settings.collection, points=points, wait=True)

    def _run(self, query: str, settings: RetrieverSettings) -> str:
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
        
        print("Choosen Settings:", settings)

        if not isinstance(settings, RetrieverSettings):
            retriever_settings = RetrieverSettings(**settings)
        else:
            retriever_settings = settings
            # print("WARNING: Using default RetrieverSettings!")
        
        best_chunks = self.__hybrid_search(
            settings=retriever_settings,
            query=query,
        )
        # Process the best_chunks to generate a response including the source
        return "\n\n".join(
            [
                f"[Source: {doc.payload.get('source', 'Unknown')}]\n{doc.payload.get('text', '')}"
                for doc in best_chunks
            ]
        )
