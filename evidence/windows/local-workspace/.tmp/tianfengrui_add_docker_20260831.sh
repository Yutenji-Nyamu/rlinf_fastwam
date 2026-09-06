set -eu

echo '== identity =='
hostname
id

echo '== targets =='
getent passwd tianfengrui >/dev/null || { echo 'ERROR: user tianfengrui does not exist'; exit 42; }
getent group docker >/dev/null || { echo 'ERROR: group docker does not exist'; exit 43; }
getent group sudo >/dev/null || { echo 'ERROR: group sudo does not exist'; exit 44; }

echo '== before =='
id tianfengrui
getent group docker
getent group sudo

sudo -S -p '' usermod -aG docker,sudo tianfengrui

echo '== after =='
id tianfengrui
getent group docker
getent group sudo
