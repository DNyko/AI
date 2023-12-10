from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names:{}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data array: {}".format(type(iris_dataset['data'])))
print("Shape of data array:{}".format(iris_dataset['data'].shape))

# Зчитуємо дані
data = iris_dataset['data']
# Виведення значень ознак для перших п'яти прикладів
print("First five rows of data array:\n{}".format(data[:5]))

print("Type of target array:{}".format(type(iris_dataset['target'])))
print("Targets:\n{}".format(iris_dataset['target']))
