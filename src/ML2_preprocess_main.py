from ML2_Asg_Pipeline import preprocess

def main():
    df_2011 = preprocess("data/day_2011.csv")
    df_2012 = preprocess("data/day_2012.csv")

    print("Preprocessing complete.")
    print("2011 shape:", df_2011.shape)
    print("2012 shape:", df_2012.shape)

if __name__ == "__main__":
    main()
