#!/bin/bash

docker-compose exec certbot certbot certonly --webroot -w /var/www/certbot -d dev.on-village.com --email lstlove9804@naver.com --agree-tos --no-eff-email

