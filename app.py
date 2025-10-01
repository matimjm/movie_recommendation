import gradio as gr
import os
import torch

from recommendation_engine import recommendation_engine
from timeit import default_timer as timer
from typing import Tuple, Dict

title = "Movie Recommendation BERT"

description = "Bidirectional encoder representations from transformers (BERT) leverages the created embeddings of over 1.2M movies from IMDB database to provide accurate recommendations based on comparing the with movie embeddings user input embeddgings using the cosine similarity."

demo = gr.Interface(
    fn=recommendation_engine,
    inputs=gr.Text(type="text"),
    outputs=[
        gr.Dataframe(label="Recommendations",show_row_numbers=True, headers=["Recommended titles"], col_count=1),
        gr.Number(label="Recommendation time (s)")
    ],
    title=title,
    description=description
)

demo.launch()