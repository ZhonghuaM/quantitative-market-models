FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE requirements-lock.txt ./
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY tests ./tests

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-lock.txt -e ".[dev]"

CMD ["python", "scripts/run_analysis.py"]
