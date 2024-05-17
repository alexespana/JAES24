# Automatic prediction of continuous integration outcome in modern software development
## Motivation :sparkles: and Problem Statement :warning:
In the context of modern software development, Continuous Integration (CI) is a widely adopted practice that seeks to automate the process of integrating code changes into a project. Despite offering numerous advantages, implementing it entails significant costs that must be addressed to ensure long-term efficiency.

The Continuous Integration phase can be costly in terms of both computational and economic resources, leading large companies like Google and Mozilla to invest millions of dollars in their CI systems. Various approaches have emerged to reduce the cost associated with computational load by avoiding certain builds that are expected not to fail. However, these approaches are not precise, sometimes making erroneous predictions that omit failed builds, leading to studies aimed at increasing the accuracy with which failed builds are predicted.

In addition to the costs associated with the computational and economic burden of CI, another issue faced by software development teams is the time they must wait to receive feedback on the outcome of the CI process. This waiting time can sometimes be significant and can negatively affect team productivity and efficiency, as well as the ability to respond quickly to problems and make adjustments in development.

This study is framed within modern software development, specifically in the realm of Continuous Integration and the automatic prediction of its outcome. The need to reduce costs associated with CI without compromising the quality and security of the process is particularly important in this area. Therefore, it proposes a thorough study of different implementations of prediction algorithms in the automatic prediction of CI outcomes, extracting various features and studying their importance in the final result.

## Objectives :dart:
The main objective of this Master's Final Project is to develop a predictive algorithm that utilizes artificial intelligence to predict whether a specific commit will pass the continuous integration phase in a project hosted on GitHub. To achieve this goal, the following sub-objectives are proposed:

- OB-01: Implement a machine learning algorithm that generates a predictive model (a predictor) based on a set of features extracted from commits.
- OB-02: Utilize the GitHub API to obtain relevant data about the commits, such as their history, associated characteristics, and previous continuous integration results.
- OB-03: Develop and implement different prediction algorithms with the selection of different features aiming to provide multiple options for predicting the outcome of continuous integration.
- OB-04: Implement a mechanism to extract human-understandable information from the predictions made by the artificial intelligence algorithms.
- OB-05: Develop a web application to analyze and present visualizations to the user about the factors influencing the prediction made by the algorithm.

This last objective will help us focus on how to present the results in a way that is understandable and useful for end users, which will be crucial for the success and adoption of the application.

In order to structure and guide the research process, research questions are established that the study aims to answer:

- PI-01: What characteristics of the commits and which prediction algorithm produce the best results in the automatic prediction of the continuous integration outcome?
- PI-02: What form of presenting the results of automatic predictions of the continuous integration outcome is most valued by users?

## Installation procedure :wrench:
### Tune-up :gear:
To run the project, you need to have Docker installed on your machine. You can download and install Docker Desktop from the following link: [Docker Desktop](https://www.docker.com/products/docker-desktop)

After installing Docker, you must clone the repository to your machine using the following command:

```bash
git clone https://github.com/alexespana/TFM.git
```

Then, navigate to the project directory and execute the following command to create the necessary environment files:

```bash
cd TFM/project
```

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```
This commands will create the environment files for the backend and frontend services. You can modify the values of the environment variables in these files to suit your needs.

__NOTE__: To use the project, you need to have a GitHub account and create a personal access token. You can create a token by following the steps described in the [official documentation](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token). After creating the token, you must add it to the `backend/.env` file in the `GITHUB_TOKEN` variable.


### Installation :hammer:
To install the project, you must execute the following commands:

```bash
chmod +x install.sh
```

```bash
./install.sh install
```

This script will create the necessary Docker images and containers to run the project. It will also install the required dependencies and start the services. After the installation process is complete, you can access the web application by opening a web browser and navigating to the following URL:

```bash
http://localhost
```

#### Other commands:
To reinstall the project, you can execute the following script:

```bash
sudo ./install.sh reinstall
```

To uninstall the project, you can execute the following script:

```bash
sudo ./install.sh uninstall
```
