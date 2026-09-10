# Makefile for Customer Support Ingestion & Brand Filtering Pipeline

BRAND ?= AppleSupport
INPUT ?= data/raw/twcs.csv

.PHONY: all install ingest test clean

all: ingest

install:
	python -m pip install -r requirements.txt

ingest:
	python run_pipeline.py --brand $(BRAND) --input $(INPUT)

test:
	pytest tests/ -v

clean:
	python -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['data/processed', 'reports', '__pycache__', '.pytest_cache']]"
