from typing import List, Tuple, Sequence, Union, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import spmatrix

def make_vectorizer(method: str = 'tfidf', max_features: int = 5000) -> Union[TfidfVectorizer, CountVectorizer]:
    """
    Create and return a text vectorizer based on specified method.
    
    Args:
        method (str): Vectorization method ('tfidf' or 'count')
        max_features (int): Maximum number of features to extract
        
    Returns:
        Union[TfidfVectorizer, CountVectorizer]: Configured vectorizer
        
    Raises:
        ValueError: If method is not 'tfidf' or 'count'
    """
    if method.lower() not in ['tfidf', 'count']:
        raise ValueError("method must be either 'tfidf' or 'count'")
    
    common_params = {
        'max_features': max_features,
        'stop_words': 'english',
        'min_df': 2,  # Ignore terms that appear in less than 2 documents
        'max_df': 0.95  # Ignore terms that appear in more than 95% of documents
    }
    
    if method.lower() == 'tfidf':
        return TfidfVectorizer(**common_params)
    else:
        return CountVectorizer(**common_params)

def build_dtm(texts: Sequence[str], vectorizer: Union[TfidfVectorizer, CountVectorizer]) -> Tuple[Union[np.ndarray, spmatrix], np.ndarray]:
    """
    Build document-term matrix from texts using provided vectorizer.
    
    Args:
        texts (Sequence[str]): List of text documents to vectorize
        vectorizer: Fitted or unfitted vectorizer instance
        
    Returns:
        Tuple[Union[np.ndarray, spmatrix], np.ndarray]: Document-term matrix and feature names
    """
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer.get_feature_names_out()

def run_lda(dtm: Union[np.ndarray, spmatrix], num_topics: int = 5, random_state: int = 42) -> LatentDirichletAllocation:
    """
    Run Latent Dirichlet Allocation on document-term matrix.
    
    Args:
        dtm (Union[np.ndarray, spmatrix]): Document-term matrix
        num_topics (int): Number of topics to extract
        random_state (int): Random seed for reproducibility
        
    Returns:
        LatentDirichletAllocation: Fitted LDA model
    """
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=random_state,
        learning_method='batch',
        max_iter=20,
        n_jobs=-1  # Use all available cores
    )
    lda.fit(dtm)
    return lda

def top_keywords_per_topic(
    lda_model: LatentDirichletAllocation,
    feature_names: np.ndarray,
    n_top: int = 10
) -> List[List[str]]:
    """
    Extract top keywords for each topic from LDA model.
    
    Args:
        lda_model: Fitted LDA model
        feature_names: List of feature names corresponding to columns in DTM
        n_top (int): Number of top keywords to extract per topic
        
    Returns:
        List[List[str]]: List of top keywords for each topic
    """
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get indices of top n_top words
        top_indices = np.argsort(-topic)[:n_top]  # Negative to sort in descending order
        topics.append([str(feature_names[i]) for i in top_indices])
    return topics

def get_document_topics(
    lda_model: LatentDirichletAllocation,
    dtm: Union[np.ndarray, spmatrix],
    threshold: float = 0.3
) -> List[List[int]]:
    """
    Get dominant topics for each document.
    
    Args:
        lda_model: Fitted LDA model
        dtm (Union[np.ndarray, spmatrix]): Document-term matrix
        threshold (float): Minimum probability threshold for topic assignment
        
    Returns:
        List[List[int]]: List of topic indices for each document
    """
    doc_topics = lda_model.transform(dtm)
    document_topics = []
    
    for doc_topic_dist in doc_topics:
        # Convert to list for consistent handling
        topics = []
        # Get topics above threshold
        for idx, prob in enumerate(doc_topic_dist):
            if prob >= threshold:
                topics.append(idx)
        # If no topics above threshold, get the highest probability topic
        if not topics:
            topics = [int(np.argmax(doc_topic_dist))]
        document_topics.append(topics)
    
    return document_topics 