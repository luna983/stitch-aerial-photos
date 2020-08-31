FROM python:3.7
RUN git clone https://github.com/luna983/stitch-aerial-photos /home/github
WORKDIR /home/github
RUN pip install -r requirements.txt
RUN pip install jupyterlab
RUN pip install flake8
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
