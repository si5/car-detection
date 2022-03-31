FROM python:3.8-slim-bullseye
RUN apt update -y \
  && apt install -y libglib2.0-0 libsm6 libxrender1 libxext6 curl
COPY requirements.txt /home/
COPY . /home/
WORKDIR /home
RUN pip install -r /home/requirements.txt
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
