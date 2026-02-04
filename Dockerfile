# Base légère Python
FROM python:3.11-slim

# Évite prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Install dépendances système utiles
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Timezone (US market handled in code, but keep UTC clean)
ENV TZ=UTC

# Dossier app
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer libs Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Créer dossiers data s’ils n’existent pas
RUN mkdir -p data/prices_cache \
    data/features_cache \
    data/backtest_reports \
    data/audit_reports

# Port Streamlit
EXPOSE 8501

# Lancement par défaut (dashboard)
CMD ["streamlit", "run", "dashboards/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
