# scripts/pubmed_fetch.py
from Bio import Entrez
import pandas as pd

Entrez.email = 'your_email@example.com'  # Add your email here

def search(query):
    handle = Entrez.esearch(db='pubmed', sort='relevance', retmax='100', retmode='xml', term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids)
    results = Entrez.read(handle)
    return results

def get_pubmed_data(queries):
    studiesIdLists = []
    for query in queries:
        studies = search(query)
        studiesIdLists.append(studies['IdList'])

    article_details = []
    for id_list in studiesIdLists:
        details = fetch_details(id_list)
        article_details.append(details)

    article_data = []
    for detail in article_details:
        for article in detail['PubmedArticle']:
            try:
                title = article['MedlineCitation']['Article']['ArticleTitle']
                abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                journal = article['MedlineCitation']['Article']['Journal']['Title']
                article_data.append({'Title': title, 'Abstract': abstract, 'Journal': journal})
            except KeyError:
                continue

    df = pd.DataFrame(article_data)
    return df

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    queries = ["trauma epigenetics"]
    df = get_pubmed_data(queries)
    save_to_csv(df, 'data/pubmed_articles.csv')
