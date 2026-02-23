# ============================================
# IB Gateway Headless - Docker Image
# ============================================
# Runs IB Gateway with IBC controller + Xvfb
# for 24/7 headless operation on Linux servers.
#
# Based on: IB Gateway 10.30+ / IBC 3.19+
# ============================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    xvfb \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxslt1.1 \
    libgtk-3-0 \
    libxxf86vm1 \
    socat \
    netcat-openbsd \
    procps \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- JRE (IB Gateway ships its own, but IBC needs system java) ----
RUN apt-get update && apt-get install -y \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# ---- IB Gateway installation ----
# Download latest stable IB Gateway (offline installer)
ARG IB_GATEWAY_VERSION=10.30
ARG IB_GATEWAY_URL=https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh

RUN curl -sSL ${IB_GATEWAY_URL} -o /tmp/ibgateway-install.sh \
    && chmod +x /tmp/ibgateway-install.sh \
    && yes '' | /tmp/ibgateway-install.sh -q -dir /opt/ibgateway \
    && rm /tmp/ibgateway-install.sh

# ---- IBC installation ----
ARG IBC_VERSION=3.19.0
ARG IBC_URL=https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip

RUN mkdir -p /opt/ibc \
    && curl -sSL ${IBC_URL} -o /tmp/ibc.zip \
    && unzip -o /tmp/ibc.zip -d /opt/ibc \
    && chmod +x /opt/ibc/*.sh \
    && rm /tmp/ibc.zip

# ---- Config directories ----
RUN mkdir -p /root/Jts /opt/ibc/config /opt/scripts

# ---- Copy configuration files ----
COPY ibc/config.ini /opt/ibc/config/config.ini
COPY ibc/jts.ini /root/Jts/jts.ini
COPY scripts/run.sh /opt/scripts/run.sh
COPY scripts/healthcheck.sh /opt/scripts/healthcheck.sh

RUN chmod +x /opt/scripts/*.sh

# ---- Environment defaults ----
# These are overridden by .env / docker-compose
ENV IBKR_USER=""
ENV IBKR_PASS=""
ENV TRADING_MODE="paper"
ENV IBC_INI="/opt/ibc/config/config.ini"
ENV IBC_PATH="/opt/ibc"
ENV TWS_PATH="/opt/ibgateway"
ENV TWS_SETTINGS_PATH="/root/Jts"
ENV DISPLAY=:99

# IB Gateway API ports
EXPOSE 4001 4002

# Healthcheck: verify IB Gateway API port is listening
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD /opt/scripts/healthcheck.sh

# ---- Start ----
CMD ["/opt/scripts/run.sh"]
