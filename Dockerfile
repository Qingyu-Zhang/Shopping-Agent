FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY shopping_agent ./shopping_agent
COPY fixtures ./fixtures
RUN pip install --no-cache-dir .
ENTRYPOINT ["python", "-m", "shopping_agent"]
