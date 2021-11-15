FROM pytorch/pytorch

RUN apt-get update 
RUN pip install albumentations pandas

RUN mkdir -p submission
ADD src submission

CMD [ "python", "./submission/test.py" ]

