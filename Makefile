.PHONY: install train test clean lint infer docker-build

# ── Environment ──────────────────────────────────────────────
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# ── Install ──────────────────────────────────────────────────
install: $(VENV)/.installed

$(VENV)/.installed: requirements.txt pyproject.toml
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	touch $(VENV)/.installed

# ── Training ─────────────────────────────────────────────────
train: install
	$(PYTHON) scripts/train.py --config configs/base_config.yaml

train-quick: install
	$(PYTHON) scripts/train.py --config configs/base_config.yaml --epochs 3

# ── Testing ──────────────────────────────────────────────────
test: install
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Inference ────────────────────────────────────────────────
infer: install
	$(PYTHON) -c "from src.inference import BioNERPipeline; p = BioNERPipeline(); print(p.predict('Patient with colorectal cancer'))"

# ── Quality ──────────────────────────────────────────────────
lint: install
	$(PYTHON) -m ruff check src/ tests/ scripts/

format: install
	$(PYTHON) -m ruff format src/ tests/ scripts/

# ── Clean ────────────────────────────────────────────────────
clean:
	rm -rf $(VENV)
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf models/*
	rm -rf logs/*
	find . -name '*.pyc' -delete

# ── Docker ───────────────────────────────────────────────────
docker-build:
	docker build -t biomed-ner-biobert .

docker-run:
	docker run --gpus all -p 8000:8000 biomed-ner-biobert