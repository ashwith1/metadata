# Metadata Extraction and RAG System
## Intelligent Document Processing with LangChain, LangGraph, and Qdrant

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.48-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple?style=for-the-badge)

An advanced document intelligence system that combines metadata extraction, semantic search, and conversational AI using LangChain, LangGraph, and Qdrant vector database. This system processes PDF documents, extracts structured metadata, and enables intelligent question-answering through Retrieval-Augmented Generation (RAG).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [System Components](#system-components)
- [Configuration](#configuration)
- [API Integration](#api-integration)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Future Enhancements](#future-enhancements)

## Overview

This project implements a complete document intelligence pipeline that:

1. **Ingests Documents**: Extracts text from PDFs using PyMuPDF and OCR
2. **Chunks Content**: Intelligently splits documents into semantic chunks
3. **Generates Embeddings**: Creates vector representations using multiple embedding models
4. **Stores Vectors**: Persists embeddings in Qdrant vector database
5. **Extracts Metadata**: Uses LLM chains to extract structured information
6. **Enables RAG**: Provides conversational Q&A over documents
7. **Traces Execution**: Monitors performance with LangFuse

### Key Capabilities

- **Multi-Model Support**: Switch between local, Together AI, and OpenAI models
- **Intelligent Chunking**: Context-aware document segmentation
- **Metadata Extraction**: Automated extraction of key information
- **Semantic Search**: Vector-based document retrieval
- **Conversational AI**: Natural language Q&A over documents
- **Offline Mode**: FAISS fallback for local deployment
- **Observability**: Complete execution tracing with LangFuse

## Features

### Document Processing
- **PDF Extraction**: Text extraction with PyMuPDF
- **OCR Support**: Image-based PDF processing with Tesseract
- **Smart Chunking**: Semantic-aware text segmentation
- **Language Detection**: Automatic language identification
- **Batch Processing**: Handle multiple documents simultaneously

### Vector Search
- **Qdrant Integration**: Production-grade vector database
- **FAISS Fallback**: Local vector storage for offline use
- **Multiple Embeddings**: Support for various embedding models
- **Hybrid Search**: Combine vector and keyword search
- **Similarity Metrics**: Configurable distance metrics

### LLM Integration
- **LangChain Framework**: Modular LLM orchestration
- **LangGraph Workflows**: End-to-end RAG pipelines
- **Multiple Providers**: OpenAI, Together AI, local models
- **Prompt Engineering**: Optimized prompts for metadata extraction
- **Chain Composition**: Complex multi-step reasoning

### User Interface
- **Streamlit Web App**: Interactive document upload and chat
- **Model Selection**: Runtime model switching
- **Chat History**: Persistent conversation context
- **Document Preview**: Inline document viewing
- **Result Export**: Download extracted metadata and answers

### Monitoring & Tracing
- **LangFuse Integration**: Complete execution tracing
- **Performance Metrics**: Latency and cost tracking
- **Error Logging**: Comprehensive error reporting
- **Debug Mode**: Detailed execution logs

## Architecture

### System Pipeline

```
┌─────────────┐
│   PDF Doc   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│  Ingestion (extractor.py)           │
│  - PyMuPDF text extraction          │
│  - Tesseract OCR for images         │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│  Chunking (chunker.py)              │
│  - Semantic text segmentation       │
│  - Overlap handling                 │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│  Embeddings (embedder.py)           │
│  - Local: sentence-transformers     │
│  - Together AI embeddings           │
│  - OpenAI embeddings                │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│  Vector Store                       │
│  - Qdrant (production)              │
│  - FAISS (fallback)                 │
└──────┬──────────────────────────────┘
       │
       ├──> Metadata Extraction
       │    (metadata_chain.py)
       │
       └──> RAG Q&A
            (rag_chain.py)
```

### LangGraph Workflow

```
Start
  │
  v
┌────────────────┐
│  Load Document │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Extract Chunks │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Generate       │
│ Embeddings     │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Store Vectors  │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Query Retrieved│
│ Context        │
└───────┬────────┘
        │
        v
┌────────────────┐
│ Generate       │
│ Response       │
└───────┬────────┘
        │
        v
      End
```

## Tech Stack

### Core Frameworks
- **LangChain 0.2.16**: LLM application framework
- **LangChain Community 0.2.16**: Community integrations
- **LangChain Core 0.2.43**: Core abstractions
- **LangGraph 0.0.55**: Workflow orchestration
- **LangFuse 2.10.0**: Observability and tracing

### Vector Database
- **Qdrant 1.7.0**: Production vector database
- **FAISS 1.7.4**: Local vector search fallback

### LLM Providers
- **OpenAI**: GPT models via API
- **Together AI**: Open-source model hosting
- **Local Models**: Sentence Transformers

### Document Processing
- **PyMuPDF 1.23.0**: PDF text extraction
- **pytesseract 0.3.10**: OCR capabilities
- **pypdf**: Alternative PDF processing
- **Pillow**: Image handling

### Web Framework
- **Streamlit 1.48.0**: Interactive web application

### Utilities
- **pandas 2.1.0**: Data manipulation
- **python-dotenv 1.0.0**: Environment configuration
- **langdetect 1.0.9**: Language identification

## Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Docker**: For Qdrant database (recommended)
- **Tesseract OCR**: For image-based PDF processing
- **Git**: Version control

### System Requirements

**Minimum**:
- 8GB RAM
- 4 CPU cores
- 10GB disk space

**Recommended**:
- 16GB RAM
- 8 CPU cores
- 20GB disk space
- GPU (for local embedding models)

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/ashwith1/metadata.git
cd metadata
```

#### 2. Create Virtual Environment

**Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install Tesseract OCR

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS**:
```bash
brew install tesseract
```

**Windows**:
Download from: https://github.com/UB-Mannheim/tesseract/wiki

#### 5. Setup Qdrant Database

**Using Docker** (Recommended):
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v ${PWD}/qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

**Windows**:
```bash
docker run -p 6333:6333 -p 6334:6334 -v %cd%\qdrant_data:/qdrant/storage qdrant/qdrant:latest
```

**Using Docker Compose**:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

#### 6. Configure Environment Variables

Copy the example configuration:
```bash
cp config/env.example .env
```

Edit `.env` and add your API keys:
```bash
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Together AI
TOGETHER_API_KEY=your_together_key_here

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LangFuse (optional)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Quick Start

### Launch the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Usage

1. **Upload a PDF**: Click "Browse files" and select your document
2. **Select Model**: Choose embedding and LLM models
3. **Extract Metadata**: Click "Extract Metadata" button
4. **Ask Questions**: Type questions in the chat interface
5. **View Results**: See extracted metadata and RAG responses

### Example Workflow

```python
# Upload: research_paper.pdf
# Model: OpenAI GPT-4
# Metadata Extracted:
# - Title: "Neural Networks for NLP"
# - Authors: ["John Doe", "Jane Smith"]
# - Date: "2024-01-15"
# - Keywords: ["NLP", "Transformers", "BERT"]

# Question: "What is the main contribution of this paper?"
# Answer: "The paper introduces a novel attention mechanism..."
```

## Usage Guide

### Web Interface (Streamlit)

#### Document Upload

1. Navigate to the sidebar
2. Click "Choose a PDF file"
3. Select one or more PDFs
4. Wait for upload confirmation

#### Model Configuration

**Embedding Models**:
- Local: `sentence-transformers/all-MiniLM-L6-v2`
- Together AI: `togethercomputer/m2-bert-80M-8k-retrieval`
- OpenAI: `text-embedding-3-small`

**LLM Models**:
- OpenAI: `gpt-4`, `gpt-3.5-turbo`
- Together AI: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Local: HuggingFace models

#### Metadata Extraction

1. Upload document
2. Click "Extract Metadata"
3. View extracted fields:
   - Title
   - Authors
   - Date/Year
   - Abstract/Summary
   - Keywords/Topics
   - Document type

#### RAG Question Answering

1. Type question in chat input
2. System retrieves relevant chunks
3. LLM generates answer with context
4. View source citations

### Programmatic Usage

```python
from backend.ingestion.extractor import PDFExtractor
from backend.ingestion.chunker import TextChunker
from backend.embeddings.embedder import EmbeddingModel
from backend.vector_store.qdrant_store import QdrantStore
from backend.llm.rag_chain import RAGChain

# Extract text from PDF
extractor = PDFExtractor()
text = extractor.extract("document.pdf")

# Chunk text
chunker = TextChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(text)

# Generate embeddings
embedder = EmbeddingModel(model_name="local")
embeddings = embedder.embed(chunks)

# Store in Qdrant
store = QdrantStore(collection_name="documents")
store.add_vectors(chunks, embeddings)

# Query with RAG
rag = RAGChain(store=store, llm="openai")
response = rag.query("What is this document about?")
print(response)
```

## System Components

### 1. Ingestion Module (`backend/ingestion/`)

#### `extractor.py`
Extracts text from PDF documents:
- **PyMuPDF**: Fast text extraction from native PDFs
- **OCR**: Tesseract for image-based PDFs
- **Preprocessing**: Text cleaning and normalization

#### `chunker.py`
Splits documents into manageable chunks:
- **Semantic Chunking**: Preserve context boundaries
- **Overlap**: Maintain continuity between chunks
- **Size Control**: Configurable chunk sizes

### 2. Embeddings Module (`backend/embeddings/`)

#### `embedder.py`
Generates vector embeddings:
- **LangChain Wrapper**: Unified interface
- **Batch Processing**: Efficient embedding generation
- **Caching**: Avoid redundant computations

#### `model_registry.py`
Manages embedding models:
- **Local Models**: sentence-transformers
- **Together AI**: Cloud-hosted models
- **OpenAI**: Commercial embeddings
- **Dynamic Switching**: Runtime model selection

### 3. Vector Store Module (`backend/vector_store/`)

#### `qdrant_store.py`
Qdrant database integration:
- **Collection Management**: Create and manage collections
- **Vector Operations**: Add, search, delete vectors
- **Filtering**: Metadata-based filtering
- **LangChain Compatible**: Seamless integration

#### `faiss_fallback.py`
Local FAISS alternative:
- **Offline Mode**: No internet required
- **Fast Search**: Optimized similarity search
- **Persistence**: Save and load indexes

### 4. LLM Module (`backend/llm/`)

#### `metadata_chain.py`
Metadata extraction chain:
- **Structured Output**: JSON-formatted metadata
- **Prompt Templates**: Optimized extraction prompts
- **Validation**: Schema validation for extracted data

#### `rag_chain.py`
Retrieval-Augmented Generation:
- **Retrieval**: Semantic search for context
- **Generation**: LLM-based answer synthesis
- **Citation**: Source attribution

### 5. Graphs Module (`backend/graphs/`)

#### `rag_graph.py`
LangGraph workflow:
- **State Management**: Track pipeline state
- **Conditional Routing**: Dynamic workflow paths
- **Error Handling**: Graceful failure recovery

#### `metadata_graph.py`
Metadata extraction workflow:
- **Multi-Step Processing**: Sequential extraction
- **Validation**: Quality checks
- **Fallback**: Alternative extraction methods

#### `monitor.py`
LangFuse integration:
- **Trace Logging**: Complete execution traces
- **Metrics**: Performance and cost tracking
- **Debugging**: Detailed error information

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Together AI Configuration
TOGETHER_API_KEY=...
TOGETHER_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=  # Optional for cloud Qdrant

# LangFuse Configuration (Optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Application Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_RETRIEVAL_RESULTS=5
TEMPERATURE=0.7
```

### Qdrant Configuration

Edit `config/qdrant.yaml`:

```yaml
host: localhost
port: 6333
grpc_port: 6334
collection:
  name: documents
  vector_size: 384  # Depends on embedding model
  distance: Cosine
  on_disk: false
```

## API Integration

### OpenAI

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### Together AI

```python
from langchain_together import Together, TogetherEmbeddings

# LLM
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# Embeddings
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    together_api_key=os.getenv("TOGETHER_API_KEY")
)
```

### Local Models

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Development

### Project Structure

See [file structure.txt](file%20structure.txt) for complete directory layout.

### Development Container

The project includes a `.devcontainer` configuration for VS Code:

```bash
# Open in VS Code
code .

# Reopen in container
# Ctrl+Shift+P -> "Dev Containers: Reopen in Container"
```

### Code Quality

**Linting**:
```bash
pip install ruff
ruff check .
```

**Formatting**:
```bash
ruff format .
```

**Type Checking**:
```bash
pip install mypy
mypy backend/
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/unit/

# Run with coverage
pytest --cov=backend tests/unit/

# Run specific test
pytest tests/unit/test_embeddings.py
```

### Integration Tests

```bash
# Test with Qdrant
pytest tests/integration/test_qdrant.py

# Test RAG pipeline
pytest tests/integration/test_rag_pipeline.py
```

### Test Coverage

```bash
pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build
docker build -t metadata-rag .

# Run
docker run -p 8501:8501 --env-file .env metadata-rag
```

### Cloud Deployment

**Streamlit Cloud**:
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy from repository

**AWS/GCP/Azure**:
- Use container services (ECS, Cloud Run, Container Apps)
- Set environment variables
- Configure Qdrant connection

## Troubleshooting

### Common Issues

**1. Qdrant Connection Error**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker restart <qdrant_container_id>
```

**2. Tesseract Not Found**
```bash
# Ubuntu
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

**3. Memory Issues**
```bash
# Reduce chunk size in config
CHUNK_SIZE=256

# Use smaller embedding model
MODEL=all-MiniLM-L6-v2
```

**4. API Rate Limits**
- Use local models for development
- Implement rate limiting
- Cache embeddings

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
streamlit run app.py
```

## Performance

### Benchmarks

**Document Processing**:
- PDF extraction: ~1-2 seconds per page
- OCR: ~5-10 seconds per page
- Chunking: ~0.1 seconds per document

**Embedding Generation**:
- Local (CPU): ~2-5 seconds per 100 chunks
- Together AI: ~1-2 seconds per 100 chunks
- OpenAI: ~0.5-1 seconds per 100 chunks

**Vector Search**:
- Qdrant: <100ms for 1M vectors
- FAISS: <50ms for 100K vectors

**RAG Query**:
- End-to-end: ~2-5 seconds
- Retrieval: ~100-200ms
- Generation: ~1-4 seconds

### Optimization Tips

1. **Batch Processing**: Process documents in batches
2. **Caching**: Cache embeddings and LLM responses
3. **Async Operations**: Use async for I/O operations
4. **Index Tuning**: Optimize Qdrant HNSW parameters
5. **Model Selection**: Balance quality vs. speed

## Future Enhancements

- [ ] Multi-modal support (images, tables)
- [ ] Advanced metadata schemas
- [ ] Custom fine-tuned embeddings
- [ ] GraphRAG implementation
- [ ] Real-time document monitoring
- [ ] Multi-user authentication
- [ ] API endpoint deployment
- [ ] Advanced analytics dashboard
- [ ] Document versioning
- [ ] Collaborative annotations

## License

This project is licensed under the MIT License.

## Acknowledgments

- **LangChain**: For the LLM orchestration framework
- **Qdrant**: For the vector database
- **Streamlit**: For the web framework
- **OpenAI**: For GPT models
- **Together AI**: For open-source model hosting

## Contact

For questions or collaboration:
- GitHub: [@ashwith1](https://github.com/ashwith1)
- Repository: [metadata](https://github.com/ashwith1/metadata)

---

**Project Status**: Active Development

**Last Updated**: December 2025

**Version**: 1.0.0
