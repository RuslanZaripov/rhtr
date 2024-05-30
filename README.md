# RHTR

# Move files into remote directory

- clone directory and move to it using `cd`

```bash
cd ~/rhtr
```

- run wsl command

```bash
wsl
```

- copy files with rsync using prepared `./exclude-list.txt`

```bash
# project_folder is a directory on a remote machine where code will be stored
rsync -chavzP --stats -e "ssh -p ${port}" . --exclude-from='./exclude-list.txt' ${remote_user}@${remote_host_or_ip}:${project_folder}/rhtr
rsync -chavzP --stats -e "ssh -p ${port}" ./data/handwritten_text_images ${remote_user}@${remote_host_or_ip}:${project_folder}/rhtr/data
rsync -chavzP --stats -e "ssh -p ${port}" ./models/segmentation/${weights_filename} ${remote_user}@${remote_host_or_ip}:${project_folder}/rhtr/models/segmentation
```

- in order to move weights inside worker container specify necessary path in the Dockerfile `rhtr/Dockerfile`

# Setup

- run and stop service by using commands below

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
