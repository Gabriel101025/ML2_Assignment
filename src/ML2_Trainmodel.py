import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ML2_Asg_Pipeline import preprocess

def main():
    df = preprocess("data/day_2011.csv")
    X = df.drop(columns=["cnt"])
    y = df["cnt"]
    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Train my preferred model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    #Technically best model
    joblib.dump(rf, "model.pkl")
    print("Model trained and saved as model.pkl")

if __name__ == "__main__":
    main()
