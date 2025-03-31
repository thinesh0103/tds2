from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import pandas as pd
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QA pipeline with SMALLER model
try:
    logger.info("Loading question-answering model...")
    qa_pipeline = pipeline(
        "question-answering",
        model="mrm8488/bert-tiny-finetuned-squadv2",  # Only 17MB!
        tokenizer="mrm8488/bert-tiny-finetuned-squadv2",
        device=-1  # Force CPU usage
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    qa_pipeline = None  # Allows health check to pass but API will return 503

class AnswerResponse(BaseModel):
    answer: str

# Context for Data Science tools questions
DS_TOOLS_CONTEXT = """
Pandas is a Python library for data manipulation and analysis, featuring DataFrame objects. 
NumPy provides support for large multi-dimensional arrays and matrices. 
Scikit-learn is a machine learning library with tools for classification, regression, and clustering. 
Matplotlib is a 2D plotting library, while Seaborn provides statistical visualizations. 
TensorFlow and PyTorch are deep learning frameworks. Jupyter Notebook is an interactive computing environment.
"""

async def process_uploaded_file(file: UploadFile):
    """Handle CSV/Excel file uploads"""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        if file_ext == '.csv':
            df = pd.read_csv(tmp_path)
        elif file_ext in ('.xlsx', '.xls'):
            df = pd.read_excel(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        return df
    
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="File processing failed")
    
    finally:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)

@app.post("/api/")
async def answer_question(
    question: str = Form(..., description="The assignment question"),
    file: UploadFile = File(None, description="Optional data file (CSV/Excel)")
):
    if not qa_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded - service unavailable")
    
    try:
        if file:
            # Process file-based questions
            df = await process_uploaded_file(file)
            context = f"""
            Dataset columns: {', '.join(df.columns)}.
            First 3 rows: {df.head(3).to_dict()}.
            """
        else:
            # Use Data Science tools context for theoretical questions
            context = DS_TOOLS_CONTEXT
        
        result = qa_pipeline(question=question, context=context)
        return AnswerResponse(answer=result['answer'])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Question processing failed")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": qa_pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
