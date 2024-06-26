import os
import json
import time
import requests
import datetime
from typing import Tuple
from utils import replace_fields, get_month_start_end, get_builds_folder
from constants import GET_PRS, GET_PR, GET_PR_COMMITS, GET_PR_FILES, GET_BUILDS, GET_BUILD, RETRY_TIME, GET_COMMIT
from flask import jsonify

class GithubManager:

    def __init__(self):
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
            "X-GitHub-Api-Version": "2022-11-28"
        }   

    def get_pull_requests(self, owner: str, repo_name: str, label: str = None, number_of_prs: int = 101) -> list:
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
                try:
                    response = requests.get(next_page, params=params, headers=self.headers)
                    response.raise_for_status()

                    data = response.json()
                    results.extend(data)

                    # Check if there's a next page
                    if 'next' in response.links:
                        next_page = response.links['next']['url']
                    else:
                        next_page = None
                except Exception:
                    time.sleep(RETRY_TIME)

            return results
        else:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return jsonify({"error": "Not found"}), 404


    def get_pull_request(self, owner: str, repo_name: str, pull_request_number: int) -> dict:
        url = replace_fields(GET_PR, owner=owner, repo_name=repo_name, pull_number=str(pull_request_number))
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({"error": "Not found"}), 404

    def get_pull_request_commits(self, owner: str, repo_name: str, pull_request_number: int, number_of_commits: int = 101) -> list:
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
                try:
                    response = requests.get(next_page, params=params, headers=self.headers)
                    response.raise_for_status()

                    data = response.json()
                    results.extend(data)

                    # Check if there's a next page
                    if 'next' in response.links:
                        next_page = response.links['next']['url']
                    else:
                        next_page = None
                except Exception:
                    time.sleep(RETRY_TIME)

            return results
        else:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                return jsonify({"error": "Not found"}), 404

    def get_pull_request_files(self, owner: str, repo_name: str, pull_request_number: int, number_of_files: int = 101) -> list:
        """
        This methods allows obtaining the files that were modified in a pull request.
        Note: Responses include a maximum of 3000 files. The paginated response returns 30 files 
        per page by default.

        Args:
        owner (str): The owner of the repository.
        repo_name (str): The name of the repository.
        pull_request_number (int): The number of the pull request.

        Returns:
        dict: The dictionary containing the files metadata.
        """
        maximum_value = 100
        url = replace_fields(GET_PR_FILES, owner=owner, repo_name=repo_name, pull_number=str(pull_request_number))
        params = {
            "per_page": number_of_files,
        }

        if number_of_files > maximum_value:
            results = []

            next_page = url

            while next_page:
                try:
                    response = requests.get(next_page, params=params, headers=self.headers)
                    response.raise_for_status()

                    data = response.json()
                    results.extend(data)

                    # Check if there's a next page
                    if 'next' in response.links:
                        next_page = response.links['next']['url']
                    else:
                        next_page = None
                except Exception:
                    time.sleep(RETRY_TIME)

            return results
        else:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return [jsonify({"error": "Not found"}), 404]

    def get_builds(self, owner: str, repo_name: str, branch: str = "main", number_of_builds: int = 101) -> dict:
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
        date = datetime.datetime.now()
        current_date = date.strftime("%Y-%m-%d")
        url = replace_fields(GET_BUILDS, owner, repo_name)

        params = {
            "per_page": number_of_builds,
            "branch": branch,
            "created": '<=' + current_date,
        }

        if number_of_builds > maximum_value:
            return self._get_all_builds(url, params)
        else:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                return jsonify({"error": "Not found"}), 404

    def _get_all_builds(self, url: str, params: dict) -> dict:
        more_builds = True
        next_page = url
        last_created_at = None

        results = {
            "total_count": None,
            "workflow_runs": []
        }

        while more_builds:
            if last_created_at:
                params["created"] = '<=' + last_created_at.split('T')[0]

            results, more_builds, last_created_at = self._get_page_builds(next_page, params, results)

        return results

    def _get_all_month_builds(self, url: str, params: dict) -> dict:
            more_builds = True
            next_page = url
            last_created_at = None
            initial_range = params["created"]
            start_date, _ = initial_range.split('..')

            results = {
                "total_count": None,
                "workflow_runs": []
            }

            while more_builds:
                if last_created_at:
                    # Rest one millisecond to avoid duplicates
                    last_created_at = datetime.datetime.strptime(last_created_at, "%Y-%m-%dT%H:%M:%SZ") - datetime.timedelta(seconds=1)
                    # Log last created_at date
                    params["created"] = start_date + '..' + last_created_at.strftime("%Y-%m-%dT%H:%M:%SZ")

                results, more_builds, last_created_at = self._get_page_builds(next_page, params, results)

            
            results["total_count"] = len(results["workflow_runs"])

            return results

    def _get_page_builds(self, page: str, params: dict, results: dict) -> Tuple[dict, bool, str]:
        count = 0
        max_iterations = 10
        more_builds = True
        last_created_at = None

        while page:
            try:
                response = requests.get(page, params=params, headers=self.headers)
                response.raise_for_status()

                data = response.json()

                if "workflow_runs" in data and data["workflow_runs"]:
                    last_created_at = data["workflow_runs"][-1]["created_at"]
                    results["workflow_runs"].extend(data["workflow_runs"])

                # Check if there's a next page
                if 'next' in response.links:
                    page = response.links['next']['url']
                else:
                    page = None
                    if count < max_iterations:
                        more_builds = False

                count += 1
            except Exception:
                time.sleep(RETRY_TIME)

        return results, more_builds, last_created_at

    def get_build(self, owner: str, repo_name: str, run_id: int) -> dict:
        url = replace_fields(GET_BUILD, owner=owner, repo_name=repo_name, run_id=str(run_id))
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({"error": "Not found"}), 404

    def save_month_builds(self, month: int, year: int, owner: str, repo_name: str, branch: str = "main") -> None:
        """
        This method allows obtaining all builds from a repository in a specific month.

        Args:
        month (int): The month to search for builds.
        year (int): The year to search for builds.
        owner (str): The owner of the repository.
        repo_name (str): The name of the repository.
        branch (str): The branch to search for builds.

        Returns:
        None
        """
        start_date, end_date = get_month_start_end(year, month)
        url = replace_fields(GET_BUILDS, owner, repo_name)

        params = {
            "per_page": 100,
            "branch": branch,
            "created": start_date + '..' + end_date
        }

        builds = self._get_all_month_builds(url, params)
        
        # Check if there are builds to avoid saving files with no builds
        if builds["total_count"] > 0:
            # Save the builds in a file, (JSON)
            with open(get_builds_folder(repo_name, branch) + start_date + '_' + end_date, "w") as f:
                f.write(json.dumps(builds))

    def get_commit(self, owner: str, repo_name: str, sha: str) -> dict:
        """
        Get a specific commit.

        Args:
        owner (str): The owner of the repository.
        repo_name (str): The name of the repository.
        sha (str): The commit sha.

        Returns:
        dict: The commit information.
        """
        url = replace_fields(GET_COMMIT, owner=owner, repo_name=repo_name, sha=sha)
        while(url):
            try:
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()

                return response.json()
            except Exception:
                time.sleep(RETRY_TIME)
