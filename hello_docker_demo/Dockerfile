# IMPORTING ANOTHER IMAGE
FROM jupyter/scipy-notebook
RUN pip install joblib
WORKDIR /usr/src/hello_docker
COPY hello_docker.py ./hello_docker.py
RUN python3 hello_docker.py