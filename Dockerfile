FROM python:3.12
WORKDIR /stt-docker
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

RUN apt-get update && apt-get install -y locales \
    && locale-gen ru_RU.UTF-8 \
    && update-locale LANG=ru_RU.UTF-8

ENV LANG ru_RU.UTF-8
ENV LANGUAGE ru_RU:ru
ENV LC_ALL ru_RU.UTF-8

COPY . .
#EXPOSE 5000

#CMD ["python", "app.py"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
