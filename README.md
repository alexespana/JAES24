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
