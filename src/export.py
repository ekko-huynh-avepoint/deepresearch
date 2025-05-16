import sys
import os
from src.knowledge_storm.reports import generate_research_report

def main():
    # Get the absolute path to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src
    
    # Build the correct path - "src" is already included in the path
    results_dir = os.path.join(project_root, "src", "results", "groq", "Covid_19")
    
    # For absolute clarity, you can also use the direct path
    direct_path = r"C:\Users\ekko.huynh\OneDrive - AvePoint\Desktop\DeepResearch\src\results\groq\Covid_19"
    
    # Check if the path exists and use it if available
    if os.path.exists(direct_path):
        print(f"Using direct path: {direct_path}")
        results_dir = direct_path
    elif os.path.exists(results_dir):
        print(f"Using constructed path: {results_dir}")
    else:
        print(f"Both paths not found. Will try one more alternative.")
        alt_path = os.path.join(os.getcwd(), "results", "groq", "Covid_19")
        if os.path.exists(alt_path):
            print(f"Found alternative path: {alt_path}")
            results_dir = alt_path
        else:
            print(f"All attempted paths failed. Please check file structure.")
            print(f"Current working directory: {os.getcwd()}")
            sys.exit(1)
    
    pdf_path = generate_research_report(results_dir, "Covid_19")
    print(f"PDF generated: {pdf_path}")

if __name__ == "__main__":
    main()