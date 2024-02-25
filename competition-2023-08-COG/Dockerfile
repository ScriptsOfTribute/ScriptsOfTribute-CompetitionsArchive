FROM ubuntu:22.04

RUN mkdir tot
COPY setup.sh /tot

WORKDIR /tot

RUN chmod u+x setup.sh && \
    ./setup.sh

RUN git clone https://github.com/ScriptsOfTribute/ScriptsOfTribute-GUI.git && \
    git clone https://github.com/ScriptsOfTribute/ScriptsOfTribute-Core.git

# ta część miała odpowiadać za zaciągniecie ostatnie builda GUI do kontenera, ale chyba to nie będzie nam potrzebne
RUN curl -s https://api.github.com/repos/ScriptsOfTribute/ScriptsOfTribute-GUI/releases/latest | \
    jq -r '.assets[] | select(.name | test("Linux")) | .browser_download_url' | \
    xargs -I {} curl -L {} -o Linux.zip && \
    unzip Linux.zip