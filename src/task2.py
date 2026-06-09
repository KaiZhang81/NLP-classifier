from pathlib import Path
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

file_path = Path(__file__).resolve().parent / "FinancialPhraseBank-v1.0" / "Sentences_50Agree.txt"
df = pd.read_csv(file_path, sep='@', names=['Text', 'Sentiment'], encoding='latin-1')

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
NUM_TOKEN = 'numtoken'

# identify noun, verb, adjective, adverb
def to_wordnet_pos(tb_tag: str) -> str:
    if tb_tag.startswith('V'): return 'v'
    if tb_tag.startswith('J'): return 'a'
    if tb_tag.startswith('R'): return 'r'
    return 'n'

def preprocess(text: str) -> list[str]:
    # 1. lowercase
    text = text.lower()

    # split letter-digit combos like "eur3" -> "eur 3"
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)

    # 2. tokenize                      
    tokens = word_tokenize(text)                     
    
    processed = []
    for tok in tokens:
        if re.match(r'^\d+([.,]\d+)?%?$', tok):
            # 3. normalize numbers
            processed.append(NUM_TOKEN)              
        elif re.match(r'^[a-z]+(-[a-z]+)*$', tok):
            # 4. keep alpha only (drops punctuation)
            processed.append(tok)                    
    
    # 5. lemmatize
    tagged = pos_tag(processed)
    processed = [lemmatizer.lemmatize(w, to_wordnet_pos(t)) for w, t in tagged]
    # drop stray single chars 
    processed = [t for t in processed if len(t) > 1]          
    
    return processed

df['Tokens'] = df['Text'].astype(str).apply(preprocess)
df['Processed_Text'] = df['Tokens'].apply(' '.join)

# save as json file 
df.to_json('processed.json', orient='records', lines=True)

# before/ after check (10 examples)
for i in df.sample(10, random_state=42).index:
    print('BEFORE:', df.loc[i, 'Text'])
    print('AFTER :', df.loc[i, 'Processed_Text'])
    print('---')