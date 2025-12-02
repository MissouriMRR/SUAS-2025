FROM docker.io/ardupilot/ardupilot-dev-base

# INSTALL DEPENDENCIES

RUN apt install -y software-properties-common && apt update && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y tmux iproute2
RUN apt-get install python3-wxgtk4.0 -y --no-install-recommends

# ARDUPILOT

RUN git clone --recurse-submodules https://github.com/ArduPilot/ardupilot

WORKDIR ardupilot

ARG ARDU_BRANCH=Copter-4.5
RUN git checkout $ARDU_BRANCH

RUN git submodule update --init --recursive

RUN ./waf configure --board sitl && ./waf copter

RUN pip install mavproxy geocoder matplotlib numpy opencv-python
RUN pip install -U numpy

# add GolfCourse location to ArduPilot locations
RUN echo '# Multirotor Locations\nGolfCourse=37.9490953,-91.7848293,0,0' >> /ardupilot/Tools/autotest/locations.txt

COPY ./sim_start_drones.sh /ardupilot/Tools/autotest/

# Environment Variables
ENV OUT_PORT=14550
ENV OUT_HOST=127.0.0.1
ENV NUM_DRONES=1

CMD /ardupilot/Tools/autotest/sim_start_drones.sh $NUM_DRONES $OUT_PORT $OUT_HOST 