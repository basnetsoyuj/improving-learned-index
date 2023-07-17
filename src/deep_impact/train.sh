#!/bin/bash

torchrun --standalone --nproc_per_node=gpu -m src.deep_impact.train --triples_path /hdd1/home/soyuj/deep-impact/qidpidtriples.train.small.tsv --queries_path /hdd1/home/soyuj/deep-impact/queries.train.tsv --collection_path /hdd1/home/soyuj/deep-impact/collection-dT5q-newterms_unique.tsv --checkpoint_dir /hdd1/home/soyuj/sjb