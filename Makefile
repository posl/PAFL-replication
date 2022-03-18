# export CONTAINER_ID=`docker ps --format {{.ID}}`
export PAFL_CONTAINER_NAME=pafl_server
export PRISM_CONTAINER_NAME=prism_server
export NONE_DOCKER_IMAGES=`docker images -f dangling=true -q`
#================================================
b: ## build docker image
	docker-compose build --no-cache
ucmain: ## docker-compose up -d and connect the container
	@make u
	@make cmain
u: ## docker-compose up -d
	docker-compose up -d
cmain: ## connect pafl container
	docker exec -it $(PAFL_CONTAINER_NAME) bash
csub: ## connect prism container
	docker exec -it $(PRISM_CONTAINER_NAME) bash
d: ## docker-compose down
	docker-compose down
rmi-none: ## remove NONE images
	docker rmi $(NONE_DOCKER_IMAGES) -f
#================================================
gpu-check:	## count available GPU devices for pytorch
	python -c "import torch;print('Available GPU devices count = {}'.format(torch.cuda.device_count()))"
#================================================
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	