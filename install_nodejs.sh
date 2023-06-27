mv /usr/local/node /tmp
wget https://cdn.npmmirror.com/binaries/node/v14.21.3/node-v14.21.3-linux-x64.tar.xz
xz -d node-v14.21.3-linux-x64.tar.xz
tar -xvf node-v14.21.3-linux-x64.tar
mv node-v14.21.3-linux-x64  /usr/local/
mv /usr/local/node-v14.21.3-linux-x64/ /usr/local/node
echo 'export NODE_HOME=/usr/local/node' | tee -a /etc/profile
echo 'export PATH=$NODE_HOME/bin:$PATH' | tee -a /etc/profile
source /etc/profile
node -v
npm -v
