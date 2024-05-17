#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
VIOLET='\033[0;35m'
RESET='\033[0m'

function check_docker() {
    if ! [ -x "$(command -v docker)" ]; then
        echo -e "${RED}Error: Docker is not installed.${RESET}" >&2
        exit 1
    fi
}

function check_docker_compose() {
    if ! [ -x "$(command -v docker-compose)" ]; then
        echo -e "${RED}Error: Docker Compose is not installed.${RESET}" >&2
        exit 1
    fi
}

function install_project() {
    echo "Installing the project..."
    mkdir backend/data
    docker compose up
    echo -e "${GREEN}Project installed successfully!${RESET}"
}

function reinstall_project() {
    echo "Reinstalling the project..."
    docker compose down
    rm -rf backend/data
    docker rmi project-flask:latest
    docker rmi project-angular:latest
    docker rmi postgres
    mkdir backend/data
    docker compose up
    echo -e "${GREEN}Project reinstalled successfully!${RESET}"
}

function uninstall_project() {
    echo "Uninstalling the project..."
    docker compose down
    rm -rf backend/data
    # Remove the images
    docker rmi project-flask:latest
    docker rmi project-angular:latest
    docker rmi postgres

    echo -e "${GREEN}Project uninstalled successfully!${RESET}"
}

# Mensaje de ayuda
function help_message() {
    echo -e "${VIOLET}Usage: $0 ${RESET}{${YELLOW}install${RESET}|${YELLOW}reinstall${RESET}|${YELLOW}uninstall${RESET}}"
    echo -e "${VIOLET}Options:${RESET}"
    echo -e "${VIOLET}  install: Install the project.${RESET}"
    echo -e "${VIOLET}  reinstall: Reinstall the project.${RESET}"
    echo -e "${VIOLET}  uninstall: Uninstall the project.${RESET}"
}

# Check if Docker and Docker Compose are installed
check_docker
check_docker_compose

# Check the number of arguments
if [ $# -ne 1 ]; then
    help_message
    exit 1
fi

# Check the option
case "$1" in
    install)
        install_project
        ;;
    reinstall)
        reinstall_project
        ;;
    uninstall)
        uninstall_project
        ;;
    *)
        echo -e "Error: no valid option."
        help_message
        exit 1
        ;;
esac

exit 0
