# This is where we implement the naive bayes algo

class NaiveBayesClassifier:

    ## Constructor
    def __init__(self, X: list, y: list, attribute_names: list):
        self.X = X # m rows x n columns
        self.y = y # m row x 1 columns
        self.attribute_names = attribute_names # n row x 1 columns
        self.dataset_size = len(y) # m
        self.values = [0, 1]


    def _joint_probability(self, attribute_id: int, attribute_value: int, class_value: int) -> float:
        '''
        param:  attribute_id: attribute index
                attribute_value: value of attribute
                class_value: value of class
        returns P(x = x_i, y = y_i)
        '''
        number_of_instances = sum([1 if self.X[i][attribute_id] == attribute_value and self.y[i] == class_value
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

    def _attribute_probability(self, attribute_id: int, attribute_value: int, class_value: int = None) -> float:
        '''
        param:  attribute_id: attribute index
                attribute_value: value of attribute
                class_value: value of class
        returns P(x = x_i) if class_value == None
        returns P(x = x_i | y = y_i) otherwise
        '''
        if class_value != None:
            return self._joint_probability(attribute_id, attribute_value, class_value) / self._class_probability(class_value)
        else:
            number_of_instances = sum([1 if self.X[i][attribute_id] == attribute_value
                                        else 0
                                        for i in range(self.dataset_size)])

            return number_of_instances / self.dataset_size

    def _class_probability_by_attribute(self, class_value: int, attribute_values: list) -> float:
        '''
        param:  class_value: value of class
                attribute_values: list of values of attributes
        returns P(y = y_i | x = (x_1,..., x_n))
        '''
        probability = self._class_probability(class_value)

        for attribute_id, attribute_value in enumerate(attribute_values):
            probability *= self._attribute_probability(attribute_id, attribute_value, class_value)

        return probability

    def _argmax(self, arr: list) -> int:
        return max(range(len(arr)), key=lambda x: arr[x])

    def _predict(self, attribute_values: list) -> int:
        '''
        param:  attribute_values: list of values of features
        returns best class value
        '''
        probability = [self._class_probability_by_attribute(class_value, attribute_values)
                        for class_value in self.values]

        return self._argmax(probability)

    def predict(self, X: list) -> list:
        '''
        param:  X: attribute matrix
        returns list of class values predicted according to X
        '''
        y_predicted = [self._predict(x) for x in X]

        return y_predicted

    def print(self):
        for class_value in self.values:
            print('P(class={})={:.2f}'.format(class_value, self._class_probability(class_value)), end=' ')

            for attribute_id, attribute_name in enumerate(self.attribute_names):
                for attribute_value in self.values:
                    print('P({}={}|{})={:.2f}'
                            .format(attribute_name, attribute_value, class_value, self._attribute_probability(attribute_id, attribute_value, class_value)), end=' ')

            print()