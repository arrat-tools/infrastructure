target: build stop run exec

build:
	docker build . -t amsign_ocr

stop: 
	- docker stop amsign_ocr
	- docker rm amsign_ocr 

run:
	docker run -it -d --gpus all --shm-size=8g -v /home/arrat/sign_detection/:/project -v /home/arrat/sign_detection/data/:/data --name amsign_ocr amsign_ocr:latest

exec:
	docker exec amsign_ocr python ./am_sign_inference_ocr.py --workdir amsign_model --imgdir data/event_images
