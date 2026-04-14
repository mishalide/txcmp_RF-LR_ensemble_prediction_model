# district champ's ML predictions

his repository holds the ML pipeline designed to predict FRC match outcomes

## why is this model special?
Statbotics and TBA are great resources, and models built purely on their historical scoring data tend to land around 67-70% accuracy. we wanted to see what would happen if we layered in our own pit scouting data on top of that.
by combining driver experience ratings, auto capabilities, climb levels, and build quality scores with EPA (from Statbotics) and OPR (from TBA), our predictor hit 87.1% accuracy and an AUC of 0.947 which isn't bad

---

pipeline architecture

1. **`build_dataset.py` (data ingestion & joining)**
   - pulls team OPR data from the The Blue Alliance (TBA) API
   - pulls team EPA (Expected Points Added) from the Statbotics API
   - reads our schedule CSVs to figure out which teams are on which alliance for every qual match
   - combines OPR, EPA, and our pit scouting data into one clean training file

2. **`train_ensemble.py` (model training)**
   - since FRC regional datasets are pretty small, we use gaussian noise pertubation to generate synthetic training examples (think of it as teaching the model to handle measurement noise and human inconsistency in scouting). 
   - trains a voting ensemble of a [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model
   - evaluates performance using a strict 5-fold stratified cross-validation loop. The gaussian noise is strictly isolated to the training folds, eliminating data leakage and ensuring the 87.1% accuracy is a genuine "blind test" result!
   - outputs results in text and in an image

---

## how to Run

wouldn't really recommend it since our pitscouting data is specific BUT you can always modify it I hope I've left enough documentation but if you have questions make an issue on this repo or reach out to me 

### prereqs
make sure you have the required libraries installed:
```bash
pip install pandas scikit-learn requests numpy statbotics==3.0.0
```

### 1. build
If you update the `raw/` files with new pit scouting (IT WOULD NEED TO MATCH OURS) (github.com/mishalide/2026-React-Scout-App) or a new schedule, simply run:
```bash
python build_dataset.py
```
then you get new training data; reach out to me if you'd be willing to donate more data (🥺🥺)

### 2. train the Model
run the ensemble script to cross-validate and print out your accuracy/AUC:
```bash
python train_ensemble.py
```
check results.txt