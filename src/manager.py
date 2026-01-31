import os
import re
from datetime import datetime, timezone
from github import Github
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

class StarManager:
    def __init__(self, token):
        self.g = Github(token)
        self.user = self.g.get_user()

    def get_starred_repos(self, since=None):
        """
        Fetches starred repositories.
        If since is provided, fetches only stars created after that date.
        """
        stars = []
        # get_starred(sort='created', direction='desc') is ideal but PyGithub defaults are usually fine.
        # We verify order.
        starred_paginated = self.user.get_starred()
        
        for starred in starred_paginated:
            # starred_at is usually timezone aware (UTC) or naive depending on PyGithub version.
            # We should ensure comparison works.
            star_date = starred.starred_at.replace(tzinfo=timezone.utc) if starred.starred_at.tzinfo is None else starred.starred_at
            
            if since:
                # Ensure 'since' is also timezone aware for comparison
                since_aware = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since
                if star_date <= since_aware:
                    break
            
            stars.append(starred)
            
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
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
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
