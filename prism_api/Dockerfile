FROM python:3.7

WORKDIR /prism_api
ADD install_launch_prism.sh /prism_api/
ADD bash_setting /root/.bashrc
RUN pip install fastapi uvicorn && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install sudo -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 8000
COPY ./app /prism_api/app
ENTRYPOINT ["bash", "install_launch_prism.sh"]