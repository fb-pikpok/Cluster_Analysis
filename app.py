import streamlit as st

from src.preprocess import explode_reviews, preprocess_data
from src.embeddings import embed_reviews, reduce_dimensions_append_array
from src.extract_topic import summarize_sequential
from src.cluster import cluster_and_append, find_closest_to_centroid
from src.visualize import visualize_embeddings, plot_over_time
from src.ui import radio_filter, range_filter

