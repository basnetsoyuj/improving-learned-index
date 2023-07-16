from argparse import ArgumentParser

from .expand_filter_precomputed import construct_collection

DEFAULTS = {
    'collection': '/hdd1/home/soyuj/collection.tsv',
    'queries': 'https://huggingface.co/datasets/macavaney/d2q-msmarco-passage',
    'scores': 'https://huggingface.co/datasets/macavaney/d2q-msmarco-passage-scores-electra',
    'output': '/hdd1/home/soyuj/expanded_collection.tsv',
    'threshold': 70,
}

DESCRIPTION = 'Construct a filtered expanded collection of MS MARCO passages with queries generated from docT5query ' \
              'and filtered using query scorers like ELECTRA (doc2query--)'

parser = ArgumentParser(prog='python -m src.doc2query--', description=DESCRIPTION)
parser.add_argument('--collection', type=str, default=DEFAULTS['collection'], help='Path to MS MARCO passages dataset')
parser.add_argument('--queries', type=str, default=DEFAULTS['queries'], help='Link to docT5query generated queries')
parser.add_argument('--scores', type=str, default=DEFAULTS['scores'], help='Link to scores for generated queries')
parser.add_argument('--output', type=str, default=DEFAULTS['output'], help='Path to output collection')
parser.add_argument('--threshold', type=float, default=DEFAULTS['threshold'],
                    help='Global threshold percentile score for filtering queries')
parser.add_argument('--unique_terms_only', action='store_true', help='Inject only unique terms in expansion')

args = parser.parse_args()

# if args.threshold is a float between 0 and 1, convert it to a percentile score
if 0 <= args.threshold <= 1:
    args.threshold *= 100
elif args.threshold < 0 or args.threshold > 100:
    raise ValueError('Threshold percentile score must be between 0 and 100')

construct_collection(
    msmarco_passages_path=args.collection,
    queries_repo=args.queries,
    scores_repo=args.scores,
    output_path=args.output,
    threshold=args.threshold,
    unique_terms_only=args.unique_terms_only,
)
