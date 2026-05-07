

-----

# Autonomous AI Agent for Clinical Decision Support in Oncology

This project is a proof-of-concept implementation of an autonomous, multimodal AI agent designed to support clinical decision-making in oncology, with direct applications in radiation therapy planning and response assessment. The agent leverages a large language model as a reasoning engine to process and synthesize multimodal inputs, including clinical text, 3D NIfTI MRI scans, and a knowledge base of medical literature.

## Inspiration

This project is inspired by the framework described in the paper:

  * **"Development and validation of an autonomous artificial intelligence agent for clinical decision-making in oncology"** (Ferber et al., *Nature Cancer*, 2025).

## Architecture

The system is built as a Unified Multimodal Agent, where a single reasoning engine orchestrates a suite of specialized tools.

1.  **Reasoning Engine:** The core of the agent is a `LangChain` Agent Executor powered by Google's `Gemini 1.5 Flash` model. It deconstructs complex queries, formulates multi-step plans, and executes them by calling the necessary tools.
2.  **Toolbox:** The agent has access to a variety of tools to gather evidence from different data modalities.
3.  **Knowledge Base:** A Retrieval-Augmented Generation (RAG) system using an `EnsembleRetriever` (combining BM25 keyword search and semantic vector search) on a `Chroma` vector store.

## Features & Implemented Tools

The agent is equipped with the following capabilities:

  * **🧠 Custom Deep Learning Model:**

      * `run_segmentation_analysis`: Utilizes a custom-built 3D U-Net-style model to perform automated tumor segmentation on four MRI modalities (T1c, T1n, T2f, T2w). It precisely quantifies tumor volume, core volume, and centroid coordinates.

  * **🗺️ Anatomical Context:**

      * `brain_atlas_coordinate_tool`: Takes the millimeter coordinates from the segmentation tool and maps them to a specific anatomical region using the AAL (Automated Anatomical Labeling) brain atlas, providing crucial clinical context.

  * **📚 Evidence-Based Retrieval:**

      * `clinical_guideline_retriever_tool`: Queries the local RAG knowledge base of trusted clinical documents (e.g., NCCN guidelines) to answer questions about standard-of-care and treatment protocols.
      * `pubmed_search_tool`: Performs live searches on the PubMed database to retrieve the latest biomedical literature and clinical trial information.

  * **🔬 Simulated Tools (for demonstrating framework extensibility):**

      * `oncokb_query_tool`: Simulates querying a precision oncology database to find therapies for specific genetic mutations.
      * `histopathology_mutation_analyzer_tool`: Simulates running an AI model on a histology slide to predict mutational status.

## Project Structure

The project is organized into a modular Python package for clarity and scalability.

```
oncology-ai-agent/
├── data/
│   └── medical_papers/
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── agent_tools.py
│   ├── rag_pipeline.py
│   └── segmentation/
│       ├── __init__.py
│       ├── model.py
│       └── utils.py
├── .env
├── .gitignore
└── requirements.txt
```

## Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd oncology-ai-agent
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**

      * Create a file named `.env` in the project's root directory.
      * Add your Google API key to this file:
        ```
        GOOGLE_API_KEY="your_actual_api_key_here"
        ```

5.  **Place Data:**

      * Place your trained segmentation model weights (`.h5` file) in a local directory and update the `MODEL_WEIGHTS_PATH` in `src/agent_tools.py`.
      * Place your clinical guideline PDFs inside the `data/medical_papers/` folder.

## How to Run

1.  Ensure your virtual environment is activated.
2.  Run the following command from the project's root directory:
    ```bash
    python -m src.app
    ```
3.  Open the local URL (e.g., `http://127.0.0.1:7860`) provided in your terminal in a web browser.

## Example Usage

1.  Upload the four required NIfTI files for an initial scan.
2.  Use a complex query that requires the agent to use multiple tools. For example:
    > "Perform a full segmentation analysis on the initial scans. After finding the tumor coordinates, use the atlas tool to identify the anatomical location. Then, based on the location, search the knowledge base for potential surgical considerations."

-----
