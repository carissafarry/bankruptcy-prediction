FROM python:3.8-slim

ENV PYTHONUNBUFFERED 1

ENV TMPDIR=/var/tmp

# Prepare and make /var/tmp writable
RUN mkdir -p /var/tmp && chmod 1777 /var/tmp \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy & install dependencies
COPY ./requirements.txt /requirements.txt

# RUN apt-get update
# RUN apt-get install sudo -y build-essential \
#     python3-pip \
#     python3-dev
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get clean

# Setup app directory
RUN mkdir /src
WORKDIR /src
COPY ./src /src

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' user
USER user