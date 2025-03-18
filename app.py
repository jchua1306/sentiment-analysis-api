from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import Optional, List, Dict, Any

# Initialize the FastAPI app
app = FastAPI(title="Sentiment Analysis API", 
              description="API for analyzing sentiment in text using transformers",
              version="1.0.0")

# Load the sentiment analysis model
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Create a request model
class SentimentRequest(BaseModel):
    text: str
    return_all_scores: Optional[bool] = False

# Create a response model
class SentimentResponse(BaseModel):
    label: str
    score: float
    all_scores: Optional[List[Dict[str, Any]]] = None

@app.post("/analyze/", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze the sentiment of the input text.
    
    - **text**: The text to analyze
    - **return_all_scores**: Whether to return scores for all possible labels (default: False)
    """
    try:
        # Check if text is too short
        if len(request.text.strip()) < 3:
            raise HTTPException(status_code=400, detail="Text is too short for sentiment analysis")
            
        # Analyze sentiment
        result = sentiment_analyzer(
            request.text,
            return_all_scores=request.return_all_scores
        )
        
        # Process result
        label = result[0]['label']
        score = result[0]['score']
        
        # Create response
        response = {
            "label": label,
            "score": score
        }
        
        # Add all scores if requested
        if request.return_all_scores:
            response["all_scores"] = result[0]
            
        return response
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

# Batch processing endpoint
class BatchSentimentRequest(BaseModel):
    texts: List[str]
    return_all_scores: Optional[bool] = False

@app.post("/analyze/batch/", response_model=List[SentimentResponse])
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """
    Analyze the sentiment of multiple texts in a single request.
    
    - **texts**: List of texts to analyze
    - **return_all_scores**: Whether to return scores for all possible labels (default: False)
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided for analysis")
            
        # Analyze sentiment for all texts
        results = sentiment_analyzer(
            request.texts,
            return_all_scores=request.return_all_scores
        )
        
        # Process results
        responses = []
        for result in results:
            response = {
                "label": result['label'],
                "score": result['score']
            }
            
            # Add all scores if requested
            if request.return_all_scores:
                response["all_scores"] = result
                
            responses.append(response)
            
        return responses
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Batch sentiment analysis failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use /analyze/ endpoint to analyze text sentiment."}

# If running the file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)