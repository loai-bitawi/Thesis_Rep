
class default_settings ():
    def __init__(self):
        self.main_path='E:/GitHub/Thesis/'
    def importer(self):
        from config import default_settings
        from joblib import Parallel, delayed, parallel_backend
        from queue import Queue
        from rank_bm25 import BM25Okapi
        from scipy.spatial.distance import pdist, squareform
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        from sklearn.feature_extraction.text import TfidfVectorizer        
        from tqdm import tqdm 
        from transformers import BertModel
        from transformers import BertTokenizer
        import concurrent.futures
        import json
        import math
        import matplotlib.pyplot as plt
        import multiprocessing
        import numpy as np
        import os
        import pandas as pd
        import random
        import re
        import spacy
        import sys
        import time
