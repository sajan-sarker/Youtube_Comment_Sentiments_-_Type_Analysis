import os
import warnings
import psycopg2
from requests import Request
import load_dotenv
import pandas as pd
from typing import Optional

from datetime import datetime
from fastapi import FastAPI, templating, HTTPException
from pydantic import BaseModel, HttpUrl
from psycopg2.extras import RealDictCursor

from scripts.utils import *
from app.utils import clean_comment, download_comments
from app.loader import model_setup
from app.ModelPredictor import ModelPredictor
from app.Plots import PlotResult

warnings.filterwarnings("ignore")
load_dotenv.load_dotenv()

app = FastAPI()

class CommentRequest(BaseModel):
    comment: str

class URLRequest(BaseModel):
    url: HttpUrl

@app.on_event('startup')
def startup_devent():
    print("Starting up...")
    global conn, cursor
    global model, tokenizer, sentiment_decoder, type_decoder, device, model_predictor, plots

    # connect to postgresql db
    try:
        conn = psycopg2.connect(
            host=os.getenv("PS_HOST"),
            database=os.getenv("PS_DB"),
            user=os.getenv("PS_USER"),
            password=os.getenv("PS_PASSWORD"),
            cursor_factory=RealDictCursor
        )
        cursor = conn.cursor()
        print("Database connection successful")
    except Exception as e:
        print("Database connection failed")
        print(e)

    os.makedirs('app/plots', exist_ok=True)

    model, tokenizer, sentiment_decoder, type_decoder, device = model_setup()
    model_predictor = ModelPredictor(model, tokenizer, device)
    plots = PlotResult()

@app.get("/")
def home():
    return {"message": "Welcome to the YouTube Comment Sentiment and Type Analysis API"}
    # return templating.Jinja2Templates(directory="template").TemplateResponse("index.html", {"request": {}})

@app.post('/predict')
def predict(comment: CommentRequest):
    comment = clean_comment(comment.comment)
    sentiment, comment_type, senti_proba, type_proba = model_predictor.get_predict(str(comment))
    sentiment = str(sentiment_decoder.classes_[int(sentiment)])
    comment_type = str(type_decoder.classes_[int(comment_type)])
    senti_proba = float(round((senti_proba*100), 2))
    type_proba = float(round((type_proba*100), 2))
    date = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    query = """
    INSERT INTO results 
    (result_type, comment, sentiment, type_, sentiment_confidence, type_confidence, date)
    VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    values = ('single', comment, sentiment, comment_type, senti_proba, type_proba, date)

    try:    
        cursor.execute(query, values)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database insert failed: {str(e)}"
        )

    return {
        "sentiment": sentiment,
        "comment_type": comment_type,
        "sentiment_confidence": senti_proba,
        "type_confidence": type_proba
    }

@app.post('/analyze')
def analyze(url: URLRequest):
    # scrape the comments from the given youtube video
    comments = download_comments(str(url.url))
    # create a dataframe to hold the comments predictions
    df = pd.DataFrame()

    # get predictions for each comment and store in the dataframe
    sentiments, types, senti_proba, type_proba = model_predictor.get_predict(comments['comment'].tolist())
    df['sentiment'] = sentiments
    df['type'] = types
    df['senti_proba'] = senti_proba
    df['type_proba'] = type_proba

    
    # mapping the predicted classes and types with its names
    df['sentiment'] = df['sentiment'].astype(int).map(lambda x: sentiment_decoder.classes_[x])
    df['type'] = df['type'].astype(int).map(lambda x: type_decoder.classes_[x])

    plots_dir = 'app/plots'
    plots_path = {
        'sd' : f"{plots_dir}/{plots.generate_plot_name('sd')}",
        'sp' : f"{plots_dir}/{plots.generate_plot_name('sp')}",
        'td' : f"{plots_dir}/{plots.generate_plot_name('td')}",
        'tp' : f"{plots_dir}/{plots.generate_plot_name('tp')}"
    }

    # plot the predicted class distributions
    plots.plot_distribution(df, col="sentiment", title="Sentiment Distribution", xlabel="Comment Sentiments", ylabel="Number of Comments", save_path=plots_path['sd'])
    plots.plot_distribution(df, col="type", title="Type Distribution", xlabel="Comment Types", ylabel="Number of Comments", save_path=plots_path['td'])

    # plot the prediction confidence
    plots.plot_confidence(df, col="sentiment", proba_col="senti_proba", title="Model Confidence per Sentiment Class", xlabel="Confidence Score (%)", ylabel="Sentiment", save_path=plots_path['sp'])
    plots.plot_confidence(df, col="type", proba_col="type_proba", title="Model Confidence per Type Class", xlabel="Confidence Score (%)", ylabel="Type", save_path=plots_path['tp'])

    return {
        "total_comments": len(sentiments),
        "sentiment_distribution_plot": plots_path['sd'],
        "sentiment_confidence_plot": plots_path['sp'],
        "type_distribution_plot": plots_path['td'],
        "type_confidence_plot": plots_path['tp']
    }

@app.get('/history')
def get_history():
    query = "SELECT * FROM results ORDER BY date DESC LIMIT 50"
    try:
        cursor.execute(query)
        records = cursor.fetchall()
        return {
            "history": records
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database query failed: {str(e)}"
        )