FROM python:3.12
WORKDIR /stt-docker
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

COPY . .
#EXPOSE 5000

#CMD ["python", "app.py"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
