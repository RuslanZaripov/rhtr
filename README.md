# RHTR

# Start wsl

```bash
wsl
```

- move to the project directory

```bash
cd /mnt/c/Users/username/Projects/rhtr
```

- copy files using rsync

```bash
rsync -chavzP --stats -e "ssh -p 2201" . --exclude-from='./exclude-list.txt' rzaripov@api.statanly.com:/storage/rzaripov/rhtr
rsync -chavzP --stats -e "ssh -p 2201" ./data/handwritten_text_images rzaripov@api.statanly.com:/storage/rzaripov/rhtr/data
rsync -chavzP --stats -e "ssh -p 2201" ./models/segmentation/linknet-7.onnx rzaripov@api.statanly.com:/storage/rzaripov/rhtr/models/segmentation
```

# Запуск

```bash
docker compose up --build
docker compose rm --stop 
```

- attach yourself to the logs of all running services
- use Ctrl+Z or Ctrl+C to detach yourself from the log output without shutting down your running containers

```bash
docker compose logs -f -t 
```

--

```bash
docker exec -it {CONTAINER_ID} bash
```

- run locust testing

```bash
locust -f .\backend\client\locustfile.py --legacy-ui
```

```bash
docker image prune
```

```bash
docker container stats
```

# Testing

- Configure `IMAGE_DIR` and `API_URL` at the top of the file ./backend/client/client.py

1. `IMAGE_DIR` - path to the directory with images, all images will be processed
2. `API_URL` - url of the API service (e.g. http://localhost:8000)

- Run console command `python ./backend/client/client.py` 


