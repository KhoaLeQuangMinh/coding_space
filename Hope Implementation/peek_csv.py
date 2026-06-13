import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python peek_csv.py <path_to_csv>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path, index_col=0)
        
        print("\n" + "="*50)
        print(f"PEEKING INTO: {csv_path}")
        print("="*50)
        
        print("\n--- COLUMNS ---")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
            
        print("\n--- DATA ---")
        # Print with pandas default formatting which aligns columns nicely
        print(df.to_string())
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

if __name__ == "__main__":
    main()
