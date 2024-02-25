# Basics
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y rsync sudo wget curl git jq unzip

# .Net Core
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x ./dotnet-install.sh
./dotnet-install.sh --channel 7.0

export DOTNET_ROOT=$HOME/.dotnet
export PATH=$PATH:$DOTNET_ROOT

