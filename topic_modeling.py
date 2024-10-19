# scripts/topic_modeling.py
import gensim
from gensim import corpora

def build_topic_model(processed_texts, num_topics=5, passes=10):
    # Create a dictionary from the preprocessed text
    dictionary = corpora.Dictionary(processed_texts)

    # Create a corpus: a bag-of-words representation of the texts
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

    return lda_model, dictionary, corpus

def print_topics(lda_model, num_words=5):
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/preprocessed_pubmed_articles.csv')
    all_tokens = df['Processed_Abstract'].apply(eval).tolist()  # Load tokenized data as list

    # Build and print topics
    lda_model, dictionary, corpus = build_topic_model(all_tokens, num_topics=5, passes=15)
    print_topics(lda_model)
