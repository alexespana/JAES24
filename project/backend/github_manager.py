from github import Github
from utils import replace_fields
from constants import *
import requests
import os

class GithubManager:

    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.getenv("GITHUB_TOKEN")}",
            "X-GitHub-Api-Version": "2022-11-28"
        }   

    def get_pull_requests(self, owner, repo_name):
        url = replace_fields(GET_PRS, owner, repo_name)
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_pull_request(self, owner, repo_name, pull_request_number):
        url = replace_fields(GET_PR, owner=owner, repo_name=repo_name, pull_number=pull_request_number)
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_pull_request_commits(self, owner, repo_name, pull_request_number):
        url = replace_fields(GET_PR_COMMITS, owner=owner, repo_name=repo_name, pull_number=pull_request_number)
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_pull_request_files(self, owner, repo_name, pull_request_number):
        url = replace_fields(GET_PR_FILES, owner=owner, repo_name=repo_name, pull_number=pull_request_number)
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_builds(self, owner, repo_name):
        url = replace_fields(GET_BUILDS, owner, repo_name)
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None
        
    def get_build(self, owner, repo_name, run_id):
        url = replace_fields(GET_BUILD, owner=owner, repo_name=repo_name, run_id=run_id)
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None
