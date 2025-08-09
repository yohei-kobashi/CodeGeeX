#!/usr/bin/env bash
set -e

echo "=== Updating apt and installing prerequisites ==="
sudo apt-get update
sudo apt-get install -y software-properties-common curl wget build-essential

echo "=== Installing Python 3.8.12 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.8=3.8.12-1+ubuntu$(lsb_release -rs) python3.8-venv python3.8-dev
# Install pip for Python3.8
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.8 get-pip.py
rm get-pip.py

echo "=== Installing OpenJDK 18.0.2.1 ==="
sudo add-apt-repository ppa:openjdk-r/ppa -y
sudo apt-get update
sudo apt-get install -y openjdk-18-jdk
java -version

echo "=== Installing Node.js 16.x and js-md5@0.7.3 ==="
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g js-md5@0.7.3

echo "=== Installing g++-7 (C++11 support) ==="
sudo apt-get install -y g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 \
  --slave   /usr/bin/g++ g++ /usr/bin/g++-7 \
  --slave   /usr/bin/gcov gcov /usr/bin/gcov-7
# You can switch with: sudo update-alternatives --config gcc

echo "=== Installing Boost 1.71.0 ==="
sudo apt-get install -y libboost-all-dev

echo "=== Installing OpenSSL 3.0.0 ==="
sudo apt-get install -y openssl libssl-dev

echo "=== Installing Go 1.18.4 ==="
GO_VERSION=1.18.4
wget https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
rm go${GO_VERSION}.linux-amd64.tar.gz
if ! grep -q "/usr/local/go/bin" ~/.profile; then
  echo "export PATH=\$PATH:/usr/local/go/bin" >> ~/.profile
fi
source ~/.profile
go version

echo "=== Setup complete! ==="
