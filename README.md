# Multi-Stage Retrieval System

This project implements a multi-stage document retrieval system using modern NLP models. It includes two main stages:

1. **Embedding-based retrieval**: Retrieve top-k most relevant passages based on their embeddings.
2. **Cross-encoder-based reranking**: Rerank the retrieved passages using a cross-encoder model.

The system is implemented using the following tools:

- **Flask**: For creating a web-based interface.
- **SentenceTransformers**: For embedding-based retrieval.
- **Transformers**: For cross-encoder reranking.
- **BEIR**: For dataset loading and evaluation.

## Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Project Structure

```bash
/flask-app
    /modules
        ├── __init__.py                # Package initializer
        ├── embedding.py               # Embedding and retrieval logic
        ├── reranking.py               # Reranking logic
        ├── evaluation.py              # Evaluation (NDCG, etc.)
        ├── dataset_loader.py          # Dataset loading
        └── chunking.py                # Document chunking
    /templates
        └── index.html                 # Frontend HTML template
    /static
        └── style.css                  # CSS for styling the frontend
    app.py                              # Main Flask app
    requirements.txt                    # Project dependencies
    README.md                           # Project documentation
``` 
## Features

- **Multi-stage retrieval pipeline**: Combines embedding-based retrieval and cross-encoder reranking.
- **Dataset support**: Includes support for datasets like Natural Questions from BEIR.
- **Frontend interface**: A simple web interface to interact with the retrieval system.
- **Modular architecture**: Code is split into reusable modules for easy maintenance.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multi-stage-retrieval-system.git
cd multi-stage-retrieval-system
```

### 2. Set up a Python Virtual Environment (Optional)

```bash
python3 -m venv venv
source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Start the Flask Development Server

Run the Flask app using the following command:

```bash
python app.py
```

By default, the app will be available at `http://127.0.0.1:5000/`.

### 2. Open the Web Interface

Open your web browser and go to `http://127.0.0.1:5000/`. You will see a simple interface where you can enter a search query. The system will return the top reranked passages based on the query.

## Usage

1. Enter your query in the search box.
2. The system will retrieve the top-k passages based on embedding similarity.
3. The retrieved passages will then be reranked using a cross-encoder reranker.
4. The top reranked passages will be displayed along with their relevance scores.

## Dependencies

This project requires the following Python packages:

- **Flask**: Web framework used to build the user interface.
- **Sentence-Transformers**: For passage embedding and retrieval.
- **Transformers**: For reranking using cross-encoders.
- **Torch**: Required for running models from Hugging Face Transformers.
- **BEIR**: For loading and evaluating the datasets.
- **Scikit-learn**: For calculating NDCG scores.

For a complete list of dependencies, refer to the `requirements.txt` file.

```bash
Flask==2.0.1
sentence-transformers==2.2.2
transformers==4.10.3
torch==1.10.0
numpy==1.21.2
scikit-learn==0.24.2
beir==1.0.0
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
