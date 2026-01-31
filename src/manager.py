import os
import re
import requests
from datetime import datetime, timezone
from types import SimpleNamespace
from github import Github
from google import genai
from openai import OpenAI
from anthropic import Anthropic

class StarManager:
    def __init__(self, token):
        self.token = token
        self.g = Github(token)
        try:
            self.user = self.g.get_user()
            print(f"DEBUG: Authenticated as GitHub User: {self.user.login}")
        except Exception as e:
            print(f"DEBUG: Failed to get user info: {e}")
            self.user = None

    def get_starred_repos(self, since=None):
        """
        Fetches starred repositories using raw GitHub API to ensure 'starred_at' is available.
        Attributes 'starred_at' and 'repo' are required.
        """
        stars = []
        page = 1
        per_page = 30
        
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3.star+json"
        }
        
        while True:
            url = f"https://api.github.com/user/starred?per_page={per_page}&page={page}"
            try:
                print(f"DEBUG: Fetching stars page {page}...")
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                print(f"DEBUG: Page {page} returned {len(data)} items.")
            except Exception as e:
                print(f"Error fetching stars: {e}")
                # If we fail to fetch, we should probably stop or raise, but let's break for now
                break
                
            if not data:
                break
                
            for item in data:
                # item structure: {'starred_at': '...', 'repo': {...}}
                starred_at_str = item.get('starred_at')
                repo_data = item.get('repo')
                
                if not starred_at_str or not repo_data:
                    continue
                    
                star_date = datetime.strptime(starred_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                
                if since:
                    if since.tzinfo is None:
                         since = since.replace(tzinfo=timezone.utc)
                    if star_date <= since:
                        return stars # Stop processing if we reach older stars
                
                # Create a simple object structure compatible with LLMProcessor
                # LLMProcessor expects: starred.repo.full_name, starred.repo.html_url, starred.repo.description
                
                repo_obj = SimpleNamespace(
                    full_name=repo_data.get('full_name'),
                    html_url=repo_data.get('html_url'),
                    description=repo_data.get('description')
                )
                
                starred_obj = SimpleNamespace(
                    starred_at=star_date,
                    repo=repo_obj
                )
                
                stars.append(starred_obj)
            
            page += 1
            
        return stars

class MarkdownManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_last_updated(self):
        if not os.path.exists(self.file_path):
            return None
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Read first few lines to find Last Updated
            for _ in range(5):
                line = f.readline()
                match = re.search(r'Last Updated: (\d{4}-\d{2}-\d{2})', line)
                if match:
                    return datetime.strptime(match.group(1), '%Y-%m-%d').replace(tzinfo=timezone.utc)
        return None

    def parse_existing_content(self):
        """
        Parses the markdown file into a dictionary: {Category: [Lines]}
        """
        if not os.path.exists(self.file_path):
            return {}

        content_dict = {}
        current_category = None
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.rstrip()
            if line.startswith('## '):
                current_category = line[3:].strip()
                if current_category not in content_dict:
                    content_dict[current_category] = []
            elif current_category and line.strip():
                content_dict[current_category].append(line)
        
        return content_dict

    def save(self, new_data, file_header):
        """
        new_data: dict {Category: [lines]}
        file_header: str (Title)
        """
        # Parse existing first to retrieve old data (if we want to merge carefully)
        # But the requirement says "Smart Sync... Append". 
        # Actually the requirement says "기존 파일 내용에 새로운 카테고리/리스트를 병합합니다."
        
        existing_data = self.parse_existing_content()
        
        # Merge new_data into existing_data
        for category, items in new_data.items():
            if category not in existing_data:
                existing_data[category] = []
            # Prepend new items to the category? Or append?
            # Usually newer items are better on top, but "Append" was mentioned.
            # "내용을 하단에 추가(Append)" -> usually means append to file. 
            # But let's assume we want to just add them.
            # Let's put new items at the top of the category list for better visibility?
            # Or just extend. Let's extend for now as per "append" but logically for a daily curator, top is better.
            # Append new items as requested
            existing_data[category] = existing_data[category] + items

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(f"{file_header}\n")
            f.write(f"Last Updated: {today}\n\n")
            
            for category, items in existing_data.items():
                f.write(f"## {category}\n")
                for item in items:
                    f.write(f"{item}\n")
                f.write("\n")

class LLMProcessor:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
    
    def process(self, repos):
        """
        repos: list of github.Repository
        Returns: dict {Category: ["- [Name](Url): Description"]}
        """
        if not repos:
            return {}

        repo_summaries = []
        for starred in repos:
            repo = starred.repo
            desc = repo.description if repo.description else "No description"
            repo_summaries.append(f"Name: {repo.full_name}\nURL: {repo.html_url}\nDescription: {desc}")
        
        prompt = (
            "Classify the following GitHub repositories into informative categories (e.g., 'Machine Learning', 'Web Development', 'Tools'). "
            "Provide a one-sentence summary for each. "
            "Format the output strictly as:\n"
            "## Category Name\n"
            "- [Repo Name](URL): Summary\n\n"
            "Repositories:\n" + "\n---\n".join(repo_summaries)
        )

        response_text = ""
        
        try:
            if self.model_name == 'GEMINI':
                client = genai.Client(api_key=self.api_key)
                
                # Dynamic model selection
                model_id = 'gemini-1.5-flash' # Fallback
                try:
                    # Preference order for "best" model
                    preferences = [
                        'gemini-2.0-flash-exp',
                        'gemini-1.5-pro',
                        'gemini-1.5-flash',
                        'gemini-1.0-pro'
                    ]
                    
                    print("DEBUG: Fetching available Gemini models...")
                    available_models = []
                    for m in client.models.list():
                        # DEBUG: Print attributes of the first model to debug
                        if not available_models: 
                            print(f"DEBUG: Model object attributes: {dir(m)}")
                        
                        # In google-genai SDK 0.x/1.x, supported_generation_methods might be missing or different.
                        # For now, let's assume if it has 'gemini' in name, it's a candidate.
                        name = m.name.split('/')[-1]
                        available_models.append(name)
                    
                    print(f"DEBUG: Available models: {available_models}")
                    
                    found = False
                    # Check preferences first
                    for pref in preferences:
                        # Match exact or simple prefix matching
                        matches = [m for m in available_models if pref in m]
                        if matches:
                            model_id = matches[0] # Pick the first match (e.g. gemini-1.5-pro-001)
                            found = True
                            print(f"DEBUG: Selected preferred model: {model_id}")
                            break
                    
                    if not found and available_models:
                         # Fallback to first available 'gemini' model
                         gemini_models = [m for m in available_models if 'gemini' in m.lower()]
                         if gemini_models:
                             model_id = gemini_models[0]
                             print(f"DEBUG: Selected fallback model: {model_id}")
                except Exception as e:
                    print(f"DEBUG: Failed to list models, using default {model_id}: {e}")

                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt
                )
                response_text = response.text
                
            elif self.model_name == 'OPENAI':
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
                
            elif self.model_name == 'CLAUDE':
                client = Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing with {self.model_name}: {e}")
            return {}

        return self._parse_llm_response(response_text)

    def _parse_llm_response(self, text):
        data = {}
        current_category = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('## '):
                current_category = line[3:].strip()
                if current_category not in data:
                    data[current_category] = []
            elif current_category and line.startswith('- '):
                data[current_category].append(line)
        
        return data
