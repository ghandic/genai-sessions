import os
from urllib.parse import urlparse

from dotenv import load_dotenv
import typer

from ask import AskWebsite
import crawler
from pre_process import  load_domain_cache, tokenize_text_with_chunking, create_embeddings, load_scraped


load_dotenv()
app = typer.Typer()


client = AskWebsite(api_key=os.environ['OPENAI_API_KEY'])


@app.command()
def crawl(url: str):
    crawler.crawl(url)
    

@app.command()
def tokenize(url: str):
    domain = urlparse(url).netloc
    df = load_domain_cache(domain)
    tokenize_text_with_chunking(df, domain)


@app.command()
def embed(url: str):
    domain = urlparse(url).netloc
    df = load_scraped(domain)
    create_embeddings(client, df, domain)


@app.command()
def ask(url: str):
    domain = urlparse(url).netloc
    client.load(domain)
    while True:
        answer = client.answer_question(input("Ask Slalom a question: "))
        print(f"Answer: {answer}")

        
if __name__ == "__main__":
    app()