FROM mcr.microsoft.com/azureml/onnxruntime-1.6-ubuntu18.04-py37-cpu-inference:latest

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# copy requirements
COPY ./requirements.txt /usr/src/app/

# set work directory
WORKDIR /usr/src/app

RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app/

# run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
