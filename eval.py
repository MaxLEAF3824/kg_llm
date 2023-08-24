import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os


def rouge_l(text1, text2):
    