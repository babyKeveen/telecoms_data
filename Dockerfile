FROM jupyter/datascience-notebook:latest

# Remove folders baked into the base image that clash with our project structure
RUN rm -rf /home/jovyan/app /home/jovyan/pipeline /home/jovyan/work

# Copy and install dependencies at build time
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --quiet
