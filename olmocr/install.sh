apt-get update -y && apt-get install -y software-properties-common \
            && add-apt-repository -y ppa:deadsnakes/ppa \
                && apt-get -y update

apt-get update && apt-get -y install python3-apt
echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
apt-get update -y && apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

apt-get update -y && apt-get install -y --no-install-recommends \
            git \
                python3.11 \
                    python3.11-dev \
                        python3.11-distutils \
                            ca-certificates \
                                build-essential \
                                    curl \
                                        unzip

rm -rf /var/lib/apt/lists/* \
            && unlink /usr/bin/python3 \
                && ln -s /usr/bin/python3.11 /usr/bin/python3 \
                        && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
                            && pip3 install -U pip

apt-get clean

git clone https://github.com/allenai/olmocr.git
cd olmocr

pip install -e .[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

#We execute this example command, because we want olmocr to download the required model
python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/horribleocr.pdf
