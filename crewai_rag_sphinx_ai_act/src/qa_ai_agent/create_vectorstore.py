import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    VectorParams,
)

load_dotenv()

EMBEDDING_MODEL = AzureOpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
)

DOCS_PATH = Path(os.path.join(os.path.dirname(__file__), "..", "..", "rsc", "docs"))

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

def chunk_md_docs(docs: List[Document]):
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
        else:
            print(f"Skipping non-markdown document: {doc.metadata.get('source')}")

    return chunks

def recreate_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
        
    if client.collection_exists(collection_name=collection_name):
        _ = client.delete_collection(collection_name=collection_name)

    client.create_collection(
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
    client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT
    )

    # Indici keyword per filtri esatti / velocità nei filtri
    for key in ["doc_id", "source", "title", "lang"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=key,
            field_schema=PayloadSchemaType.KEYWORD
        )

def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "doc_id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "lang": doc.metadata.get("lang", "en"),
            "text": doc.page_content,
            "section": doc.metadata.get("section"),
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts

def build_qdrant_vector_store(
        client: QdrantClient,
        collection_name: str,
        embedding_model: Embeddings = EMBEDDING_MODEL,
        docs_path: str = DOCS_PATH,
    ) -> None:

    docs = load_docs(docs_path)
    chunks = chunk_md_docs(docs)

    vector_size = 1536

    recreate_collection(client, collection_name, vector_size)
    embs = embedding_model.embed_documents([c.page_content for c in chunks])
    points = build_points(chunks, embs)
    client.upsert(collection_name=collection_name, points=points, wait=True)

if __name__ == "__main__":
    build_qdrant_vector_store(
        client=QdrantClient(url="localhost:6333"),
        collection_name="test",
    )

