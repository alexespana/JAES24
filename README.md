# Automatic prediction of continuous integration outcome in modern software development

<p align="center">
<a href="https://github.com/alexespana/JAES24/actions/workflows/build.yml"><img src="https://github.com/alexespana/JAES24/actions/workflows/build.yml/badge.svg" alt="Build Images"></a>
<a href="https://github.com/alexespana/JAES24/actions/workflows/doc.yml"><img src="https://github.com/alexespana/JAES24/actions/workflows/doc.yml/badge.svg" alt="Build LaTeX documentation"></a>
<a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="Build LaTeX documentation"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png" alt="Build LaTeX documentation"></a>
</p>

## Motivation :sparkles: and Problem Statement :warning:
In the context of modern software development, Continuous Integration (CI) is a widely adopted practice that seeks to automate the process of integrating code changes into a project. Despite offering numerous advantages, implementing it entails significant costs that must be addressed to ensure long-term efficiency.

The Continuous Integration phase can be costly in terms of both computational and economic resources, leading large companies like Google and Mozilla to invest millions of dollars in their CI systems. Various approaches have emerged to reduce the cost associated with computational load by avoiding certain builds that are expected not to fail. However, these approaches are not precise, sometimes making erroneous predictions that omit failed builds, leading to studies aimed at increasing the accuracy with which failed builds are predicted.

In addition to the costs associated with the computational and economic burden of CI, another issue faced by software development teams is the time they must wait to receive feedback on the outcome of the CI process. This waiting time can sometimes be significant and can negatively affect team productivity and efficiency, as well as the ability to respond quickly to problems and make adjustments in development.

This study is framed within modern software development, specifically in the realm of Continuous Integration and the automatic prediction of its outcome. The need to reduce costs associated with CI without compromising the quality and security of the process is particularly important in this area. Therefore, it proposes a thorough study of different implementations of prediction algorithms in the automatic prediction of CI outcomes, extracting various features and studying their importance in the final result.

## Objectives :dart:
The main objective of this Master's Final Project is to develop a predictive algorithm that utilizes artificial intelligence to predict whether a specific commit will pass the continuous integration phase in a project hosted on GitHub. To achieve this goal, the following sub-objectives are proposed:
- **OB-01**: Implement a machine learning algorithm that generates a predictive model (a predictor) based on a set of features extracted from the builds.
- **OB-02**: Use the GitHub API to retrieve relevant data about the builds, such as their history, associated features, and previous continuous integration results.
- **OB-03**: Develop and implement different prediction algorithms with the selection of various features, aiming to provide multiple options when predicting the outcome of continuous integration.
- **OB-04**: Implement a graphical interface that serves as a data input point for the prediction algorithm and allows visualization of the obtained results.

In order to structure and guide the research process, research questions are established that the study aims to answer:
- **PI-01**: Which prediction algorithm produces the best results in the automatic prediction of continuous integration outcomes?
    - **Metric**: Accuracy, precision, recall, and F1-score of the model.
- **PI-02**: Which features of the builds are most significant in the prediction?
    - **Metric**: Importance of each feature through the interpretation of the model coefficients.

## Installation procedure :wrench:
### Tune-up :gear:
To run the project, you need to have Docker installed on your machine. You can download and install Docker Desktop from the following link: [Docker Desktop](https://www.docker.com/products/docker-desktop)

After installing Docker, you must clone the repository to your machine using the following command:

```bash
git clone https://github.com/alexespana/JAES24.git
```

Then, navigate to the project directory and execute the following command to create the necessary environment files:

```bash
cd JAES24/project
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

## License

This repository contains both code and documentation, each with a different license:

- **Code (in the `project` directory)**: Licensed under the [GNU General Public License v3.0 (GPL-3.0)](./LICENSE_CODE).
- **Documentation (in the `doc` directory)**: Licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](./LICENSE_DOC).

Please refer to each license file for specific terms and conditions.
