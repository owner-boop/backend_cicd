
#!/usr/bin/bash

sudo systemctl daemon-reload
sudo rm -f /etc/nginx/sites-enabled/default

sudo cp /home/ubuntu/fd_backend/nginx/nginx.conf /etc/nginx/sites-available/frauddetectionbackend
sudo ln -s /etc/nginx/sites-available/frauddetectionbackend /etc/nginx/sites-enabled/
#sudo ln -s /etc/nginx/sites-available/blog /etc/nginx/sites-enabled
#sudo nginx -t


sudo gpasswd -a www-data ubuntu
sudo systemctl restart nginx

