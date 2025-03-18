# Sentiment Analysis API

A FastAPI application that provides an API for sentiment analysis using the DistilBERT model fine-tuned on the SST-2 dataset.

## Features

- RESTful API endpoint for text sentiment analysis
- Based on the `distilbert-base-uncased-finetuned-sst-2-english` pre-trained model
- Support for batch processing of multiple texts
- Option to return scores for all sentiment classes
- Containerized with Docker for easy deployment
- Auto-generated API documentation with Swagger UI

## Requirements

- Python 3.9+
- Docker Desktop (for containerized deployment)
- Git (for version control)

## Installation

### Option 1: Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/jchua1306/sentiment-analysis-api.git
cd sentiment-analysis-api
```

2. Build and start the Docker container:
```bash
docker-compose up --build
```

### Option 2: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/jchua1306/sentiment-analysis-api.git
cd sentiment-analysis-api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn app:app --reload
```

## Usage

Once the application is running, you can access:

- API endpoint: http://localhost:8001/analyze/
- Batch API endpoint: http://localhost:8001/analyze/batch/
- API documentation: http://localhost:8001/docs

### Example API Request - Single Text

```python
import requests

url = "http://localhost:8001/analyze/"
payload = {
    "text": "This movie was fantastic! I really enjoyed the plot and the characters were well developed.",
    "return_all_scores": False
}
response = requests.post(url, json=payload)
print(response.json())
```

### Example API Request - Batch Processing

```python
import requests

url = "http://localhost:8001/analyze/batch/"
payload = {
    "texts": [
        "This movie was fantastic! I really enjoyed it.",
        "The service at this restaurant was terrible.",
        "I'm feeling neutral about this product."
    ],
    "return_all_scores": True
}
response = requests.post(url, json=payload)
print(response.json())
```

## API Parameters

### Single Text Analysis

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | string | (required) | The text to analyze |
| return_all_scores | boolean | false | Whether to return scores for all possible sentiment labels |

### Batch Text Analysis

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| texts | array of strings | (required) | The texts to analyze |
| return_all_scores | boolean | false | Whether to return scores for all possible sentiment labels |

## Project Structure

```
sentiment-analysis-api/
├── app.py              # FastAPI application code
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [DistilBERT Model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)