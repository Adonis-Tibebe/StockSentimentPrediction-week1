import pytest
import numpy as np
from src.topic_modeling import (
    make_vectorizer,
    build_dtm,
    run_lda,
    top_keywords_per_topic,
    get_document_topics
)

@pytest.fixture
def sample_texts():
    return [
        "apple new iphone launch event september",
        "tesla electric car production numbers increase",
        "google cloud services revenue growth",
        "apple iphone sales record quarter",
        "tesla factory expansion plans",
        "google ai research breakthrough"
    ]

def test_make_vectorizer():
    # Test TFIDF vectorizer
    tfidf_vec = make_vectorizer('tfidf', max_features=100)
    assert tfidf_vec.max_features == 100  # type: ignore
    assert tfidf_vec.stop_words == 'english'  # type: ignore
    
    # Test Count vectorizer
    count_vec = make_vectorizer('count', max_features=50)
    assert count_vec.max_features == 50  # type: ignore
    assert count_vec.stop_words == 'english'  # type: ignore
    
    # Test invalid method
    with pytest.raises(ValueError):
        make_vectorizer('invalid')

def test_build_dtm(sample_texts):
    vectorizer = make_vectorizer('tfidf', max_features=20)
    dtm, features = build_dtm(sample_texts, vectorizer)
    
    # Check shapes
    assert dtm.shape[0] == len(sample_texts)
    assert dtm.shape[1] == len(features)
    assert len(features) <= 20

def test_run_lda(sample_texts):
    vectorizer = make_vectorizer('tfidf', max_features=20)
    dtm, features = build_dtm(sample_texts, vectorizer)
    
    num_topics = 3
    lda = run_lda(dtm, num_topics=num_topics)
    
    # Check model properties
    assert lda.n_components == num_topics  # type: ignore
    assert lda.components_.shape == (num_topics, len(features))

def test_top_keywords_per_topic(sample_texts):
    vectorizer = make_vectorizer('tfidf', max_features=10)  # Reduced features for test
    dtm, features = build_dtm(sample_texts, vectorizer)
    
    num_topics = 2
    n_top = 3  # Reduced number of top words for test
    lda = run_lda(dtm, num_topics=num_topics)
    topics = top_keywords_per_topic(lda, features, n_top=n_top)
    
    # Check output structure
    assert len(topics) == num_topics
    assert all(len(topic) == n_top for topic in topics)
    assert all(isinstance(word, str) for topic in topics for word in topic)

def test_get_document_topics(sample_texts):
    vectorizer = make_vectorizer('tfidf', max_features=10)  # Reduced features for test
    dtm, features = build_dtm(sample_texts, vectorizer)
    
    num_topics = 2  # Reduced number of topics for test
    lda = run_lda(dtm, num_topics=num_topics)
    doc_topics = get_document_topics(lda, dtm, threshold=0.3)
    
    # Check output structure
    assert len(doc_topics) == len(sample_texts)
    assert all(isinstance(topics, list) for topics in doc_topics)
    # Check each topic index is valid
    for topics in doc_topics:
        for topic_idx in topics:
            assert isinstance(topic_idx, int)
            assert 0 <= topic_idx < num_topics 