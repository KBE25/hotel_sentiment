import pandas as pd
import spacy
import re
import emoji
from symspellpy import SymSpell, Verbosity
import pkg_resources 
import os 
import sys 

from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm 


# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
# These variables will be initialized ONCE per worker process by the 'worker_initializer' function.
# This is the most robust way to handle heavy objects like spaCy models with multiprocessing.
global_nlp = None
global_sym_spell = None
global_custom_stop_words = ["hotel", "room", "stay", "guest", "place", "reviews", "review"]


# --- Initializer Function for Multiprocessing Pool ---
# This function runs once in each child process when it starts up.
def worker_initializer():
    global global_nlp, global_sym_spell # Declare that we are modifying the global variables

    # Initialize spaCy model in each worker process
    try:
        global_nlp = spacy.load("en_core_web_sm") # Load with all components enabled for flexibility
        # print(f"[{os.getpid()}] spaCy 'en_core_web_sm' model loaded successfully.") # Optional debug print
    except OSError:
        # This part ideally won't run if spacy.cli.download was run upfront
        # but provides a fallback for individual workers.
        print(f"[{os.getpid()}] spaCy 'en_core_web_sm' model not found. Attempting to download in worker...")
        spacy.cli.download("en_core_web_sm")
        global_nlp = spacy.load("en_core_web_sm")
    
    # Add custom stop words to this worker's nlp vocab
    for word in global_custom_stop_words:
        global_nlp.vocab[word].is_stop = True

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
    # print(f"[{os.getpid()}] Worker initialization complete.") # Optional debug print


# --- Preprocessing Function ---
# This function performs all the text cleaning and transformation steps on a single text input.
# It uses the globally initialized 'global_nlp' and 'global_sym_spell' objects.
def preprocess_text_enhanced_spacy(text, apply_spell_correction=False):
    # Access the globally initialized objects within the worker process's scope
    global global_nlp, global_sym_spell 

    if not isinstance(text, str):
        return "" # Handle non-string inputs gracefully

    # --- Core Preprocessing Steps ---
    text = text.lower() # 1. Lowercasing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # 2. Remove URLs
    text = re.sub(r'<.*?>', '', text) # 2. Remove HTML tags
    
    text = emoji.demojize(text) # 3. Emoji/Emoticon Conversion
    text = re.sub(r':([a-zA-Z0-9_]+):', r'\1', text) # Clean demojized text (remove colons)
    text = text.replace('_', ' ') # Replace underscores in demojized text (e.g., smiling face)

    # Process text with spaCy (uses the global_nlp object initialized in the worker)
    doc = global_nlp(text) 

    # Prepare a list of lemmas for potential negation handling
    temp_lemmas_for_negation = [token.lemma_ for token in doc]
    
    # 4. Negation Handling
    negation_tokens = {"not", "no", "n't", "never", "hardly", "barely", "scarcely", "seldom"}
    for i, token in enumerate(doc):
        if token.lemma_ in negation_tokens:
            for j in range(i + 1, min(i + 4, len(doc))): # Look ahead 1 to 3 words
                next_token = doc[j]
                # Apply _NEG suffix if it's not punctuation, a stop word, or a number
                if not next_token.is_punct and not next_token.is_stop and not next_token.like_num:
                    temp_lemmas_for_negation[j] = next_token.lemma_ + "_NEG"
                else:
                    break # Stop if a boundary is hit
    
    # 5. Final Filtering and Spell Correction
    lemmas = []
    for i, token in enumerate(doc):
        current_lemma = temp_lemmas_for_negation[i] 

        if apply_spell_correction and current_lemma.strip() and current_lemma not in negation_tokens:
            if re.fullmatch(r'[a-zA-Z]+', current_lemma): # Ensure it's purely alphabetic for correction
                # SymSpell lookup (uses the global_sym_spell object initialized in the worker)
                suggestions = global_sym_spell.lookup(current_lemma, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    corrected_word = suggestions[0].term
                    current_lemma = corrected_word

        # Filter out punctuation, numbers, and stopwords (using spaCy's flags and custom list)
        if not token.is_punct and not token.like_num and not token.is_stop and len(current_lemma.strip()) > 1:
            if current_lemma.endswith("_NEG") or not token.is_stop: # Keep negated words even if original was a stop word
                lemmas.append(current_lemma)

    return " ".join(lemmas).strip()


# --- Helper function for parallel processing with tqdm progress bar ---
def parallelize_series_with_tqdm(series, func_to_apply, n_cores=None, **func_kwargs):
    if n_cores is None:
        n_cores = mp.cpu_count()

    # Prepare arguments for pool.starmap: each item is a tuple (text_item, apply_spell_correction_bool)
    items_with_args = [(item, func_kwargs.get('apply_spell_correction', False)) for item in series]

    results = [] # To collect results from all processes
    # Use multiprocessing.Pool for parallel execution, with the initializer function
    with mp.Pool(n_cores, initializer=worker_initializer) as pool: # <--- Pass initializer here!
        # pool.starmap applies 'func_to_apply' to each tuple in 'items_with_args'
        processed_items_iterator = pool.starmap(func_to_apply, items_with_args, chunksize=100)

        # Wrap the iterator with tqdm for the progress bar display
        for result in tqdm(processed_items_iterator, total=len(series), desc=f"Processing '{series.name}'"):
            results.append(result)
            
    # Convert the collected list of results back to a pandas Series, preserving original index and name.
    return pd.Series(results, index=series.index, name=series.name)


# --- MAIN EXECUTION BLOCK ---
# This 'if __name__ == '__main__':` block is CRUCIAL for multiprocessing to work correctly.
# All code that runs when the script is executed directly should be inside this block.
if __name__ == '__main__':
    print("Starting main script execution...")

    # --- Create a Dummy DataFrame (REPLACE WITH YOUR ACTUAL DATA LOADING) ---
    # Example: df_filtered = pd.read_csv('your_reviews_data.csv')
    # This dummy data is enlarged to better demonstrate multiprocessing benefits.
    data = {
        'raw_review_text': [
            "This place is not good, but actually awesome! The food was bad :( and the service was 💯.",
            "Friendly host, perfect breakfast! I realllly like it. Great value for money! 😊",
            "The hotel was great! No complains. Very clean. The staff was super nice.",
            "Average stay. The bed was uncomfortable. Wifi was slow.😡",
            "Excellent! Will come back soon. Best service ever.",
            "A bit noisy, but overall okay. The bathroom was small.",
            "Nice ambiance, good coffee. Love this place! :)",
            "Worst experience ever. Never again. The bed was terrible.",
            "Clean and comfortable. Recommended.",
            "Lovely staff and great location near the park."
        ] * 2000, # Example: Makes it 20,000 lines (total 20k rows)
        'review_title': [
            "Terrible experience :(",
            "Highly recommend B&B!",
            "Fantastic stay!",
            "Okay",
            "Amazing!",
            "Noise issue",
            "Coffee lover",
            "Nightmare",
            "Good",
            "Perfect"
        ] * 2000,
        'tags': [
            "business trip solo traveller junior suite stay night submit mobile",
            "leisure trip couple deluxe suite stay night submit mobile",
            "family trip standard room 2 nights online booking",
            "solo trip basic room one night web",
            "couple trip suite 3 nights mobile",
            "business trip single deluxe 1 night web",
            "leisure trip couple standard 2 nights mobile",
            "family trip basic 3 nights online",
            "solo trip standard 1 night mobile",
            "leisure trip couple suite 2 nights web"
        ] * 2000
    }
    df_filtered = pd.DataFrame(data)

    print(f"\nLoaded DataFrame with {len(df_filtered)} rows.")
    print("--- Original DataFrame Sample ---")
    print(df_filtered.head(2).to_string())
    print("-" * 50)

    # --- Apply the Enhanced Preprocessing Function to all text columns ---
    text_cols_to_process = ['raw_review_text', 'review_title', 'tags']
    processed_output_cols = [f'{col}_processed' for col in text_cols_to_process]

    for col in text_cols_to_process:
        print(f"Starting parallel processing for column: '{col}' (with full spaCy pipeline)...")
        # Call the parallelize_series_with_tqdm function for each column
        df_filtered[f'{col}_processed'] = parallelize_series_with_tqdm(
            df_filtered[col], # Pass the pandas Series for the current column
            preprocess_text_enhanced_spacy, # Pass your preprocessing function (top-level)
            n_cores=None, # Use all available CPU cores
            apply_spell_correction=True # Pass keyword arguments to your preprocessing function
        )
        print(f"Finished processing column: '{col}'")
    
    print("\n--- Processed Text Columns Sample (after all preprocessing) ---")
    print(df_filtered[processed_output_cols].head(2).to_string())
    print("-" * 50)

    