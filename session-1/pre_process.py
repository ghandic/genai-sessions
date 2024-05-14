import os
import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm

tqdm.pandas()



tokenizer = tiktoken.get_encoding("cl100k_base")


def remove_newlines(series: pd.Series) -> pd.Series:
    series = series.str.replace('\n', ' ')
    series = series.str.replace('\\n', ' ')
    series = series.str.replace('  ', ' ')
    series = series.str.replace('  ', ' ')
    return series

def load_domain_cache(domain:str) -> pd.DataFrame:
     # Create a list to store the text files
    texts=[]

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):

        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    return df
    
    


def split_into_many(text:str, max_tokens:int) -> list[str]:

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
        
    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


def tokenize_text_with_chunking(df: pd.DataFrame, domain:str, max_tokens: int = 500) -> pd.DataFrame:
    shortened = []
    df['n_tokens'] = df.text.progress_apply(lambda x: len(tokenizer.encode(x)))
    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            split = split_into_many(row[1]['text'], max_tokens)
            shortened += split
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )
            
    chunked_df = pd.DataFrame(shortened, columns = ['text'])
    chunked_df['n_tokens'] = chunked_df.text.progress_apply(lambda x: len(tokenizer.encode(x)))
    chunked_df.to_csv(f'processed/{domain}-scraped.csv')
    return chunked_df
    # chunked_df.n_tokens.hist()
    
def load_scraped(domain:str):
    return pd.read_csv(f'processed/{domain}-scraped.csv')

def create_embeddings(client:OpenAI, df:pd.DataFrame, domain:str) -> pd.DataFrame:
    df['embeddings'] = df.text.progress_apply(lambda x: client.embeddings.create(input=x, model='text-embedding-ada-002').data[0].embedding)
    df.to_csv(f'processed/{domain}-embeddings.csv')
    return df