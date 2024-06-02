# Распознавания рукописного русского текста

- Прочитать на другом язык [English](README.md)

- [Видео демонстрация распознавания](/static/video.mp4)

![Пример распознавания 1](/static/1.png)

![Пример распознавания 2](/static/2.png)

# Переместить файлы на удаленный компьютер

- клонируйте репозиторий и перейдите при помощи утилиты `cd`

```bash
cd ~/rhtr
```

- запустите wsl

```bash
wsl
```

- скопируйте файлы командой rsync, используя `./exclude-list.txt`

```bash
# project_folder - директория на удаленном сервере, где будет расположен код
rsync -chavzP --stats -e "ssh -p ${port}" . --exclude-from='./exclude-list.txt' ${remote_user}@${remote_host_or_ip}:${project_folder}/rhtr
rsync -chavzP --stats -e "ssh -p ${port}" ./data/handwritten_text_images ${remote_user}@${remote_host_or_ip}:${project_folder}/rhtr/data
rsync -chavzP --stats -e "ssh -p ${port}" ./models/segmentation/${weights_filename} ${remote_user}@${remote_host_or_ip}:${project_folder}/rhtr/models/segmentation
```

- чтобы переместить веса внутрь контейнера обработчика, нужно не забыть указать необходимый путь в
  Dockerfile `rhtr/Dockerfile`

# Запуск

- запустить и остановить сервис можно, используя команды ниже

```bash
docker compose up --build
docker compose rm --stop 
```

- чтобы видеть логи запущенных контейнеров
- используй Ctrl+Z или Ctrl+C, чтобы отсоединиться от логов, не останавливая работу сервиса

```bash
docker compose logs -f -t 
```

- чтобы открыть терминал внутри контейнера, используй команду внизу (удобно для отладки сервиса)

```bash
docker exec -it {CONTAINER_ID} bash
```

- чтобы убрать "висящие" изображения, используй команду внизу

```bash
docker image prune
```

- чтобы видеть статистику контейнера, используй

```bash
docker container stats
```

# Тестирование

- Пропиши переменные `IMAGE_DIR`, `API_URL` в файле `./backend/client/config.py`

1. `IMAGE_DIR` - путь к директории с изображениями, все изображения будут обработаны
2. `API_URL` - url для API (http://localhost:8000)

- Запусти команду `python ./backend/client/client.py`

# Нагрузочное тестирование Locust

- Также необходимо прописать переменные `IMAGE_DIR`, `API_URL`, как в секции выше
- Запустить тестирование можно командой внизу

```bash
locust -f .\backend\client\locustfile.py --legacy-ui
```

- Откройте url веб-интерфейса который написан в выводе команды
