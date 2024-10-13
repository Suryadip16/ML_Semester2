from collections import Counter
import math


def entropy(labels):
    total_count = len(labels)
    label_counts = Counter(labels)
    entropy_value = 0.0

    for label, count in label_counts.items():
        probability = count / total_count
        entropy_value -= probability * math.log2(probability)

    return entropy_value


def information_gain(parent_data, child1_data, child2_data):
    parent_entropy = entropy(parent_data)
    total_count = len(parent_data)
    child1_weight = len(child1_data) / total_count
    child2_weight = len(child2_data) / total_count

    child1_entropy = entropy(child1_data)
    child2_entropy = entropy(child2_data)

    weighted_child_entropy = child1_weight * child1_entropy + child2_weight * child2_entropy

    return parent_entropy - weighted_child_entropy


# Example usage:
parent_data = ['A', 'A', 'A', 'B', 'B', 'B', 'B']
child1_data = ['A', 'A', 'B']
child2_data = ['A', 'B', 'B', 'B']

print("Information Gain:", information_gain(parent_data, child1_data, child2_data))

# Example usage:
data_points = [1, 2, 3, 4, 5, 6]
class_labels = ['A', 'B', 'A', 'A', 'B', 'B']

print("Entropy:", entropy(class_labels))
