import os
import re
import time
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

        all_data = {}
        batch_size = 40
        
        print(f"DEBUG: Processing {len(repos)} stars in batches of {batch_size}...")
        
        for i in range(0, len(repos), batch_size):
            batch = repos[i : i + batch_size]
            print(f"DEBUG: Processing batch {i//batch_size + 1}/{(len(repos)-1)//batch_size + 1} ({len(batch)} items)")
            
            repo_summaries = []
            for starred in batch:
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

            response_text = self._generate_content(prompt)
            if not response_text:
                print(f"DEBUG: Batch {i//batch_size + 1} failed or returned empty.")
                continue
                
            batch_data = self._parse_llm_response(response_text)
            
            # Merge batch_data into all_data
            for category, items in batch_data.items():
                if category not in all_data:
                    all_data[category] = []
                all_data[category].extend(items)
            
            # Rate limit protection (Free tier is ~15 RPM, so 4s+ is needed. Using 10s to be safe)
            time.sleep(10)
                
        return all_data

    def _generate_content(self, prompt):
        try:
            if self.model_name == 'GEMINI':
                client = genai.Client(api_key=self.api_key)
                
                # Dynamic model selection
                model_id = 'gemini-1.5-flash' # Fallback
                try:
                    # Preference order
                    preferences = [
                        'gemini-2.5-flash',
                        'gemini-2.0-flash',
                        'gemini-1.5-pro',
                        'gemini-1.5-flash'
                    ]
                    
                    # Caching available models could be an optimization, but negligible for daily run
                    available_models = []
                    ignore_keywords = ['video', 'vision', 'image', 'tts', 'embedding', 'aqa', 'search', 'tool', 'robotics']
                    
                    # We might fail to list models if API key lacks permission, handle gracefully
                    for m in client.models.list():
                         name = m.name.split('/')[-1]
                         if any(k in name.lower() for k in ignore_keywords):
                             continue
                         available_models.append(name)
                    
                    found = False
                    for pref in preferences:
                        matches = [m for m in available_models if pref in m]
                        if matches:
                            matches.sort(key=len)
                            model_id = matches[0]
                            found = True
                            print(f"DEBUG: Selected preferred model: {model_id}")
                            break
                    
                    if not found and available_models:
                         gemini_models = [m for m in available_models if 'gemini' in m.lower()]
                         if gemini_models:
                             model_id = gemini_models[0]
                             print(f"DEBUG: Selected fallback model: {model_id}")
                             
                except Exception as e:
                    print(f"DEBUG: Model discovery failed, using fallback {model_id}. Error: {e}")

                # Custom Retry Logic for 429 Handling
                max_retries = 5
                base_delay = 30
                
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=model_id,
                            contents=prompt
                        )
                        return response.text
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str:
                            wait_time = base_delay * (2 ** attempt) # Exponential backoff: 30, 60, 120...
                            print(f"DEBUG: Hit Rate Limit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            # Not a rate limit error, raise or return None
                            print(f"DEBUG: API Error: {e}")
                            return None
                
                print("DEBUG: Max retries exceeded for rate limit.")
                return None
                
            elif self.model_name == 'OPENAI':
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
                
            elif self.model_name == 'CLAUDE':
                client = Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing with {self.model_name}: {e}")
            return None
        
        return None

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
