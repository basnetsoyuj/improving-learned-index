import re
from typing import List


def get_term_set(text):
    return set(re.sub(r'[^\w\s]', ' ', text).lower().split())


def get_unique_query_terms(query_list, passage):
    terms = get_term_set(passage)
    query_terms = get_term_set(' '.join(query_list))
    return query_terms.difference(terms)


def merge(document: str, queries: List[str]) -> str:
    document = document.replace('\n', ' ')
    unique_query_terms_str = ' '.join(get_unique_query_terms(queries, document))
    document = re.sub(r"\s{2,}", ' ', f'{document} {unique_query_terms_str}')
    return document
