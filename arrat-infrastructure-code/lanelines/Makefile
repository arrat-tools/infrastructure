target: build stop run exec

build: 
	docker build . -t clrnet_ra

run:
	docker run -it --name clrnet_ra -d --gpus all --shm-size=8g -v /home/arrat/lanelines/:/shared_host clrnet_ra

stop: 
	- docker stop clrnet_ra
	- docker rm clrnet_ra

exec:
	docker exec clrnet_ra "./call_clrnet.sh"
