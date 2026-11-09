.PHONY: help test fixture run run-lambda clean validate cite

SERVICE_NAME ?= suite2p
PYTHON ?= python3
export DOCKER_DEFAULT_PLATFORM = linux/amd64

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test        - pytest: the synthetic fixture through the processor, both layouts (no Docker, ~2 min)"
	@echo "make fixture     - write the synthetic input to data/fixture/{linked,flat}/"
	@echo "make run         - build the cpu image and run it in Docker over data/fixture/linked/ into data/output/"
	@echo "make run-lambda  - the same as Lambda runs it: read-only root, arbitrary uid, everything under /tmp"
	@echo "make validate    - app.yml against the manifest schema, CITATION.cff with cffconvert"
	@echo "make cite        - check every citation against Crossref/DataCite/PubMed"
	@echo "make clean       - remove data/output/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

test:
	MPLBACKEND=Agg $(PYTHON) -m pytest -q tests

fixture:
	rm -rf data/fixture
	$(PYTHON) -m tests.make_fixture data/fixture --layout both

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@find data/output -type f | sort

run-lambda: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml build
	docker run --rm --read-only --user 65534:65534 --tmpfs /tmp:size=4g --memory 4g --network none \
	  -v $(CURDIR)/data/fixture/linked:/in:ro -v $(CURDIR)/data/output:/out \
	  --entrypoint sh $(SERVICE_NAME):cpu -c 'mkdir -p /tmp/s/home /tmp/s/tmp /tmp/s/in /tmp/s/out && cp -r /in/. /tmp/s/in/ && \
	    HOME=/tmp/s/home TMPDIR=/tmp/s/tmp INPUT_DIR=/tmp/s/in OUTPUT_DIR=/tmp/s/out USE_GPU=false python3 -m processor.entry && cp -r /tmp/s/out/. /out/'
	@find data/output -type f | sort

validate:
	$(PYTHON) -c "import json, yaml, jsonschema; s = json.load(open('../workflow-systems-neuroscience/docs/app-manifest.v1.schema.json')); jsonschema.validate(yaml.safe_load(open('app.yml')), s); print('app.yml ok')"
	$(dir $(PYTHON))cffconvert --validate && echo "CITATION.cff ok"

cite:
	$(PYTHON) ../workflow-systems-neuroscience/scripts/verify-citations.py .

clean:
	rm -rf data/output
