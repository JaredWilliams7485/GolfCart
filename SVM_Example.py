import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingRegressor


#    ~other possible architectures~
#rnd_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
#ada_clf = AdaBoostClassifier(
#    DecisionTreeClassifier(max_depth=1), n_estimators=200,
#    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
#gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)


#    ~one line Random Forest~
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)

#    ~import example training data~
np_array = np.load("/Users/jaredwilliams/Documents/AI/GolfCart/out.npy")
#print(np_array.shape)

rand_input = np.random.rand(50000, 75)
rand_output = np.random.randint(2, size=50000)
#print(np_array.shape)

#fit expects a 2D array, so I need to flatten this from a nx25x3 into a nx75
np_array = np.reshape(np_array, (-1, 75))
rnd_clf.fit(rand_input, rand_output)
print(rnd_clf.predict(np_array))
