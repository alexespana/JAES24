import os
import requests
from utils import replace_fields
from constants import GET_PRS, GET_PR, GET_PR_COMMITS, GET_PR_FILES, GET_BUILDS, GET_BUILD
from flask import jsonify

class GithubManager:

    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
            "X-GitHub-Api-Version": "2022-11-28"
        }   

    def get_pull_requests(self, owner, repo_name, label=None, number_of_prs=101):
        """
        This method allows obtaining up to the last 100 pull requests from a
        repository. If the number of pull requests exceeds the maximun value (100),
        all repository pull requests will be returned.

        Args:
        owner (str): The owner of the repository.
        repo_name (str): The name of the repository.
        branch (str): The branch to search for pull requests.
        number_of_prs (int): The number of pull requests to return.

        Returns:
        dict: The dictionary containing the pull requests metadata.
        """
        maximum_value = 100
        url = replace_fields(GET_PRS, owner, repo_name)
        params = {
            "per_page": number_of_prs,
            "head": label
        }

        if number_of_prs > maximum_value:
            results = []

            next_page = url
            
            while next_page:
                response = requests.get(next_page, params=params, headers=self.headers)
                response.raise_for_status()

                data = response.json()
                results.extend(data)

                # Check if there's a next page
                if 'next' in response.links:
                    next_page = response.links['next']['url']
                else:
                    next_page = None

            return results
        else:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return jsonify({"error": "Not found"}), 404


    def get_pull_request(self, owner, repo_name, pull_request_number):
        url = replace_fields(GET_PR, owner=owner, repo_name=repo_name, pull_number=str(pull_request_number))
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({"error": "Not found"}), 404

    def get_pull_request_commits(self, owner, repo_name, pull_request_number, number_of_commits=101):
        """"
        This method allows obtaining up to the last 100 commits from a pull request. If 
        the number of commits exceeds the maximun value (100), all pull request commits
        will be returned.

        Args:
        owner (str): The owner of the repository.
        repo_name (str): The name of the repository.
        pull_request_number (int): The number of the pull request.
        number_of_commits (int): The number of commits to return.

        Returns:
        dict: The dictionary containing the commits metadata.
        """
        maximum_value = 100
        url = replace_fields(GET_PR_COMMITS, owner=owner, repo_name=repo_name, pull_number=str(pull_request_number))
        params = {
            "per_page": number_of_commits,
        }

        if number_of_commits > maximum_value:
            results = []

            next_page = url
            
            while next_page:
                response = requests.get(next_page, params=params, headers=self.headers)
                response.raise_for_status()

                data = response.json()
                results.extend(data)

                # Check if there's a next page
                if 'next' in response.links:
                    next_page = response.links['next']['url']
                else:
                    next_page = None

            return results
        else:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return jsonify({"error": "Not found"}), 404

    def get_pull_request_files(self, owner, repo_name, pull_request_number):
        url = replace_fields(GET_PR_FILES, owner=owner, repo_name=repo_name, pull_number=str(pull_request_number))
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({"error": "Not found"}), 404

    def get_builds(self, owner, repo_name, branch="main", number_of_builds=101):
        """
        This method allows obtaining up to the last 100 workflow runs from a 
        repository. If the number of workflow runs exceeds the maximun
        value (100), all repository workflow runs will be returned.

        Args:
        owner (str): The owner of the repository.
        repo_name (str): The name of the repository.
        number_of_builds (int): The number of workflow runs to return.

        Returns:
        dict: The dictionary containing the workflow runs metadata.
        """
        maximum_value = 100
        url = replace_fields(GET_BUILDS, owner, repo_name)
        params = {
            "per_page": number_of_builds,
            "branch": branch
        }

        if number_of_builds > maximum_value:
            results = {
                "workflow_runs": []
            }

            next_page = url
            
            while next_page:
                response = requests.get(next_page, params=params, headers=self.headers)
                response.raise_for_status()

                data = response.json()

                if "workflow_runs" in data:
                    results["total_count"] = data["total_count"]
                    results["workflow_runs"].extend(data["workflow_runs"])

                # Check if there's a next page
                if 'next' in response.links:
                    next_page = response.links['next']['url']
                else:
                    next_page = None

            return results
        else:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return jsonify({"error": "Not found"}), 404

    def get_build(self, owner, repo_name, run_id):
        url = replace_fields(GET_BUILD, owner=owner, repo_name=repo_name, run_id=str(run_id))
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({"error": "Not found"}), 404
