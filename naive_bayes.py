# This is where we implement the naive bayes algo

class NaiveBayesClassifier:
    def __init__(self, X: list, y: list, feature_names: list):
        self.X = X # m rows x n columns
        self.y = y # m row x 1 columns
        self.feature_names = feature_names # n row x 1 columns
        self.dataset_size = len(y) # m
        self.values = [0, 1]

    def _joint_probability(self, feature_id: int, feature_value: int, class_value: int) -> float:
        '''
        param:  feature_id: feature index
                feature_value: value of feature
                class_value: value of class
        returns P(x = x_i, y = y_i)
        '''
        number_of_instances = sum([1 if self.X[i][feature_id] == feature_value and self.y[i] == class_value 
                                    else 0
                                    for i in range(self.dataset_size)])

        return number_of_instances / self.dataset_size

    def _class_probability(self, class_value: int) -> float:
        '''
        param:  class_value: value of class
        returns P(y = y_i)
        '''
        number_of_instances = sum([1 if y == class_value
                                    else 0
                                    for y in self.y])

        return number_of_instances / self.dataset_size

    def _feature_probability(self, feature_id: int, feature_value: int, class_value: int = None) -> float:
        '''
        param:  feature_id: feature index
                feature_value: value of feature
                class_value: value of class
        returns P(x = x_i) if class_value == None
        returns P(x = x_i | y = y_i) otherwise
        '''
        if class_value != None:
            return self._joint_probability(feature_id, feature_value, class_value) / self._class_probability(class_value)
        else:
            number_of_instances = sum([1 if self.X[i][feature_id] == feature_value
                                        else 0
                                        for i in range(self.dataset_size)])

            return number_of_instances / self.dataset_size

    def _class_probability_by_features(self, class_value: int, feature_values: list) -> float:
        '''
        param:  class_value: value of class
                feature_values: list of values of features
        returns P(y = y_i | x = (x_1,..., x_n))
        '''
        probability = self._class_probability(class_value)

        for feature_id, feature_value in enumerate(feature_values):
            probability *= self._feature_probability(feature_id, feature_value, class_value)

        return probability

    def _argmax(self, arr: list) -> int:
        return max(range(len(arr)), key=lambda x: arr[x])

    def _predict(self, feature_values: list) -> int:
        '''
        param:  feature_values: list of values of features
        returns best class value
        '''
        probability = [self._class_probability_by_features(class_value, feature_values)
                        for class_value in self.values]
        
        return self._argmax(probability)

    def predict(self, X: list) -> list:
        '''
        param:  X: feature matrix
        returns list of class values predicted according to X
        '''
        y_predicted = [self._predict(x) for x in X]

        return y_predicted

    def print(self):
        for class_value in self.values:
            print('P(class={})={:.2f}'.format(class_value, self._class_probability(class_value)), end=' ')

            for feature_id, feature_name in enumerate(self.feature_names):
                for feature_value in self.values:
                    print('P({}={}|{})={:.2f}'
                            .format(feature_name, feature_value, class_value, self._feature_probability(feature_id, feature_value, class_value)), end=' ')

            print()