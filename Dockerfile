FROM python:3.9-bookworm

# Set working directory in the container
WORKDIR /root

# Install dependencies
RUN pip3 install --no-cache-dir \
    dash \
    cloudpickle \
    pandas \
    dash-bootstrap-components \
    numpy \
    scikit-learn \
    joblib\
    mlflow \
    dash

# Copy the application code
COPY . /root/

EXPOSE 8050

# Start the Dash app
CMD ["python", "app/app.py"]