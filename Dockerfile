FROM python:3.7
RUN pip install numpy
RUN git clone --recursive https://github.com/skvark/opencv-python.git
WORKDIR opencv-python
ENV CMAKE_ARGS "-DOPENCV_ENABLE_NONFREE=ON"
ENV ENABLE_CONTRIB 1
ENV ENABLE_HEADLESS 1
RUN pip wheel .
RUN pip install opencv_contrib_python_headless-4.4.0.40-cp37-cp37m-linux_x86_64.whl
RUN pip install -r requirements.txt
