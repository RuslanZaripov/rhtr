# RHTR

# Download models

- download models from here `https://disk.yandex.ru/d/rxlpAgiTJYWrjA`
- use the same folder structure like in the link above 
- make root folder `/models` and place it inside project (more convenient) 

# Move files into remote directory

- clone directory and move to it
- move to the project directory

```bash
cd /mnt/c/Users/username/Projects/rhtr
```

- run wsl command

```bash
wsl
```

- copy files using rsync and prepared `./exclude-list.txt`

```bash
rsync -chavzP --stats -e "ssh -p 2201" . --exclude-from='./exclude-list.txt' rzaripov@api.statanly.com:/storage/rzaripov/rhtr
rsync -chavzP --stats -e "ssh -p 2201" ./data/handwritten_text_images rzaripov@api.statanly.com:/storage/rzaripov/rhtr/data
rsync -chavzP --stats -e "ssh -p 2201" ./models/segmentation/linknet-7.onnx rzaripov@api.statanly.com:/storage/rzaripov/rhtr/models/segmentation
```

- remember to specify nessecary models path inside worker Dockerfile `rhtr/Dockerfile`

# Запуск

- run and stop serivce can be done by using commands below

```bash
docker compose up --build
docker compose rm --stop 
```

- attach yourself to the logs of all running services
- use Ctrl+Z or Ctrl+C to detach yourself from the log output without shutting down your running containers

```bash
docker compose logs -f -t 
```

- to open bash inside container run the following command
- useful when you want to debug application

```bash
docker exec -it {CONTAINER_ID} bash
```

- in order to remove dangling images run 

```bash
docker image prune
```

- if you want to see stats 

```bash
docker container stats
```

# Testing

- Configure `IMAGE_DIR` and `API_URL` in file `./backend/client/config.py`

1. `IMAGE_DIR` - path to the directory with images, all images will be processed
2. `API_URL` - url of the API service (e.g. http://localhost:8000)

- Run console command `python ./backend/client/client.py`

# Locust testing

- Same way as in previous section configure `IMAGE_DIR` and `API_URL`
- Run locust testing using command below

```bash
locust -f .\backend\client\locustfile.py --legacy-ui
```

- Open web interface url which was written in console output and fill in api url
