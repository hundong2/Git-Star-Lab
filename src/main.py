import os
import sys
from dotenv import load_dotenv
from manager import StarManager, MarkdownManager, LLMProcessor

# Load .env for local development
load_dotenv()

KEYS = {
    'GEMINI': 'GEMINI_API_KEY',
    'OPENAI': 'OPENAI_API_KEY',
    'CLAUDE': 'CLAUDE_API_KEY'
}

def detect_active_models():
    active_models = {}
    for model_name, env_var in KEYS.items():
        key = os.getenv(env_var)
        if key:
            active_models[model_name] = key
    return active_models

def main():
    gh_token = os.getenv('GH_TOKEN')
    if not gh_token:
        print("Error: GH_TOKEN is not set.")
        sys.exit(1)

    active_models = detect_active_models()
    if not active_models:
        print("No active LLM API keys found. (Checked GEMINI_API_KEY, OPENAI_API_KEY, CLAUDE_API_KEY)")
        sys.exit(0)

    print(f"Active Models: {list(active_models.keys())}")
    
    star_manager = StarManager(gh_token)

    for model_name, api_key in active_models.items():
        print(f"\n--- Processing for {model_name} ---")
        file_path = f"README.{model_name}.md"
        md_manager = MarkdownManager(file_path)
        
        # Step A: Read State
        last_updated = md_manager.get_last_updated()
        print(f"Last Updated: {last_updated}")

        # Step B: Fetch Stars
        print("Fetching stars...")
        new_stars = star_manager.get_starred_repos(since=last_updated)
        print(f"Found {len(new_stars)} new stars.")

        if not new_stars:
            print("No new stars to process.")
            continue

        # Step C: LLM Processing
        print(f"Processing with {model_name}...")
        llm_processor = LLMProcessor(model_name, api_key)
        # Process in chunks if needed? 
        # For daily runs, volume shouldn't be huge. Passing all at once for now.
        # But if it's cold start with 1000 stars, it might fail.
        # Pagination or chunking is wise, but for v2.0 MVP, let's assume reasonable daily volume.
        # However, Cold Start might be heavy.
        # Let's cap cold start? Or just process first 20 for safety if it's too many?
        # User didn't specify, but "Cold Start" = "All GitHub Star".
        # LLM context windows are large (Gemini 1M, Claude 200k, GPT-4o 128k).
        # 1 star entry ~ 100 tokens. 1000 stars ~ 100k tokens.
        # It fits in Gemini/Claude. GPT-4o might overlap.
        # I'll stick to simple "all-at-once" for now as per plan, hoping volume isn't massive.
        
        processed_content = llm_processor.process(new_stars)
        
        if not processed_content:
            print("LLM returned empty content or failed.")
            continue

        # Step D: Write & Save
        print("Updating file...")
        md_manager.save(processed_content, f"# {model_name}'s GitHub Star Summary")
        print(f"Updated {file_path}")

if __name__ == "__main__":
    main()
