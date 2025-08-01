import pandas as pd
import spacy
import re
import emoji
from symspellpy import SymSpell, Verbosity
import pkg_resources
import os
import sys
import multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm


# GLOBAL VARIABLES FOR WORKER PROCESSES
# These variables will be initialized ONCE per worker process by the 'worker_initializer' function.
# This is the most robust way to handle heavy objects like spaCy models with multiprocessing.
global_nlp = None
global_sym_spell = None
global_custom_stop_words = ["hotel", "room", "stay", "guest", "place", "reviews", "review",
                            "night", "trip", "submit", "mobile", "leisure", "double", "couple",
                            "traveller", "solo", "business", "day", "week", "family", "friend",
                            "time", "year", "people", "number"] # Expanded custom stop words


# Initializer Function for Multiprocessing Pool
# This function runs once in each child process when it starts up.
def worker_initializer():
    global global_nlp, global_sym_spell # Declare that we are modifying the global variables

    # Initialize spaCy model in each worker process
    try:
        global_nlp = spacy.load("en_core_web_sm") # Load with all components enabled for flexibility
    except OSError:
        # This part ideally won't run if spacy.cli.download was run upfront
        # but provides a fallback for individual workers.
        print(f"[{os.getpid()}] spaCy 'en_core_web_sm' model not found. Attempting to download in worker...")
        spacy.cli.download("en_core_web_sm")
        global_nlp = spacy.load("en_core_web_sm")
    
    # Add custom stop words to this worker's nlp vocab
    for word in global_custom_stop_words:
        global_nlp.vocab[word].is_stop = True

    # --- ADDITION FOR 'of' STOP WORD ---
    # Manually ensure 'of' is marked as a stop word, as some spaCy versions/pipelines
    # might not flag it if it has an unusual part-of-speech or context.
    # Being explicit ensures it's consistently treated as a stop word.
    global_nlp.vocab["of"].is_stop = True
    # --- END ADDITION ---

    # Initialize SymSpell in each worker process
    global_sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    try:
        # Locate and load the frequency dictionary for SymSpell
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        if not global_sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
            print(f"[{os.getpid()}] SymSpell dictionary file not found at expected path in worker. Spell correction may not work.")
    except Exception as e:
        print(f"[{os.getpid()}] Error loading SymSpell dictionary in worker: {e}. Spell correction may not work.")


# Preprocessing Function
# This function performs all the text cleaning and transformation steps on a single text input.
# It uses the globally initialized 'global_nlp' and 'global_sym_spell' objects.
def preprocess_text_enhanced_spacy(text, apply_spell_correction=False):
    # Access the globally initialized objects within the worker process's scope
    global global_nlp, global_sym_spell

    if not isinstance(text, str):
        return "" # Handle non-string inputs gracefully

    # Core Preprocessing Steps
    text = text.lower() # 1. Lowercasing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # 2. Remove URLs
    text = re.sub(r'<.*?>', '', text) # 2. Remove HTML tags
    
    text = emoji.demojize(text) # 3. Emoji/Emoticon Conversion
    text = re.sub(r':([a-zA-Z0-9_]+):', r'\1', text) # Clean demojized text
    text = text.replace('_', ' ') # Replace underscores in demojized text (e.g., smiling face)

    # Robustly remove numerical tokens and common number-like phrases
    text = re.sub(r'\b\d+(\.\d+)?(min|h|st|nd|rd|th)?\b', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)

    # Process text with spaCy (uses the global_nlp object initialized in the worker)
    doc = global_nlp(text)

    # Prepare a list of lemmas for potential negation handling
    temp_lemmas_for_negation = [token.lemma_ for token in doc]
    
    # 4. Negation Handling
    negation_tokens = {"not", "no", "n't", "never", "hardly", "barely", "scarcely", "seldom"}
    # ADDITION: Define conjunctions that break negation scope
    conjunction_breakers = {"but", "however", "although", "though", "yet"} 
    
    for i, token in enumerate(doc):
        if token.lemma_ in negation_tokens:
            for j in range(i + 1, min(i + 4, len(doc))): # Look ahead 1 to 3 words
                next_token = doc[j]
                
                # If a conjunction that breaks negation scope is found, stop applying negation
                if next_token.lemma_ in conjunction_breakers:
                    break 
                
                # Apply _NEG suffix if it's not punctuation, a stop word, or a number
                if not next_token.is_punct and not next_token.is_stop and not next_token.like_num:
                    temp_lemmas_for_negation[j] = next_token.lemma_ + "_NEG"
                else:
                    # If we hit punctuation, a stop word (that's not negated), or a number,
                    # assume the negation scope ends here.
                    break 
            
    # 5. Final Filtering and Spell Correction
    lemmas = []
    for i, token in enumerate(doc):
        current_lemma = temp_lemmas_for_negation[i]

        # Only apply spell correction if the flag is True and it's a valid word to correct
        if apply_spell_correction and current_lemma.strip() and current_lemma not in negation_tokens:
            if re.fullmatch(r'[a-zA-Z]+', current_lemma): # Ensure it's purely alphabetic for correction
                suggestions = global_sym_spell.lookup(current_lemma, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    corrected_word = suggestions[0].term
                    current_lemma = corrected_word

        # Filter out punctuation, numbers, and stopwords (using spaCy's flags and custom list)
        # This line remains the same; the change is in how 'token.is_stop' is determined for 'of'.
        # Keep negated words even if original was a stop word, and only keep lemmas longer than 1 char.
        if not token.is_punct and not token.like_num and len(current_lemma.strip()) > 1:
            # Explicitly check if it's a stop word or if it's a negated term (which we want to keep)
            if current_lemma.endswith("_NEG") or not token.is_stop: 
                lemmas.append(current_lemma)

    return " ".join(lemmas).strip()


# --- Helper function for parallel processing with tqdm progress bar ---
def parallelize_series_with_tqdm(series, func_to_apply, n_cores=None, **func_kwargs):
    if n_cores is None:
        n_cores = mp.cpu_count()

    # Pass apply_spell_correction through func_kwargs to the worker
    items_with_args = [(item, func_kwargs.get('apply_spell_correction', False)) for item in series]

    results = []
    with mp.Pool(n_cores, initializer=worker_initializer) as pool:
        # Use starmap to pass multiple arguments (text, apply_spell_correction)
        processed_items_iterator = pool.starmap(func_to_apply, items_with_args, chunksize=100)

        for result in tqdm(processed_items_iterator, total=len(series), desc=f"Processing '{series.name}'"):
            results.append(result)
            
    return pd.Series(results, index=series.index, name=series.name)