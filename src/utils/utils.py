import re


def get_term_set(text):
    return set(re.sub(r'[^\w\s]', ' ', text).lower().split())


def get_unique_query_terms(query_list, passage):
    terms = get_term_set(passage)
    query_terms = get_term_set(' '.join(query_list))
    return query_terms.difference(terms)
