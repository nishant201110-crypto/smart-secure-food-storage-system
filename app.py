from flask import *
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# load dataset
df = pd.read_csv("secure_dataset.csv")

# inputs
X = df[["Temperature_C","Humidity_%"]]

# outputs
y_months = df["Predicted_Lifespan_Months"]
y_action = df["Farmer_Action"]

# split
X_train, X_test, y_train_m, y_test_m = train_test_split(X, y_months, test_size=0.2, random_state=42)
X_train2, X_test2, y_train_a, y_test_a = train_test_split(X, y_action, test_size=0.2, random_state=42)

# model 1 (months prediction)
model_month = KNeighborsRegressor(n_neighbors=5)
model_month.fit(X_train, y_train_m)

# model 2 (farmer action prediction)
model_action = KNeighborsClassifier(n_neighbors=5)
model_action.fit(X_train2, y_train_a)

# label decoding
actions = {
0:"Cooling & ventilation + Neem leaves",
1:"Immediate cooling & inspection required",
2:"No action required",
3:"Sun drying",
4:"Sun drying + Neem leaves"
}

@app.route("/", methods=["GET","POST"])
def form():

    temperature=None
    humidity=None
    months=None
    days=None
    action=None

    if request.method=="POST":

        temperature=float(request.form["temperature"])
        humidity=float(request.form["humidity"])

        data=np.array([[temperature,humidity]])

        # predict months
        months=round(model_month.predict(data)[0],2)

        # convert months to days
        days=int(months*30)

        # predict farmer action
        action_code=model_action.predict(data)[0]
        action=actions[action_code]

    return render_template(
        "index.html",
        temperature=temperature,
        humidity=humidity,
        prediction=months,
        days=days,
        farmer_action=action
    )

if __name__=="__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))