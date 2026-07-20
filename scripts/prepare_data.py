from src.data.preprocess import prepare_data

if __name__ == "__main__":
    summary = prepare_data()
    print("Prepared graph data:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
