from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

'''
CtL : Concept-to-Label

Class for the transparent model representing a function from concepts to task labels.
Represents the decision-making process of a given black-box model, in the concept representation
'''


class CtLModel:

    def __init__(self, **params):

        # Create copy of passed-in parameters
        self.params = params

        # Assign parameter values
        self.clf_type   = self.params.get("method", "DT")
        self.n_concepts = self.params.get("n_concepts")
        self.n_classes  = self.params.get("n_classes")
        self.c_names    = self.params.get("concept_names", ["Concept_" + str(i) for i in range(self.n_concepts)])
        self.cls_names  = self.params.get("class_names", ["Class_ " + str(i) for i in range(self.n_classes)])


    def train(self, c_data, y_data):

        if self.clf_type == 'DT':
            clf = DecisionTreeClassifier(class_weight='balanced')
        elif self.clf_type == 'LR':
            clf = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
        elif self.clf_type == 'LinearRegression':
            clf = LinearRegression()
        elif self.clf_type == 'GBT':
            clf = GradientBoostingClassifier()
        else:
            raise ValueError("Unrecognised model type...")

        clf.fit(c_data, y_data)
        self.clf = clf


    def predict(self, c_data):
        return self.clf.predict(c_data)


