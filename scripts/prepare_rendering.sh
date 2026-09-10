#!/usr/bin/env bash
# System packages live outside the persistent SageMaker home volume.
set -euo pipefail
sudo -n apt-get update -qq
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libglib2.0-0t64 libnss3 libnspr4 libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libcups2t64 libdbus-1-3 libxcb1 libxkbcommon0 libatspi2.0-0t64 libx11-6 \
    libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 \
    libcairo2 libpango-1.0-0 libasound2t64 fonts-liberation
