import re
from graph import plot_metrics


def parse_log_file(file_path):
    """
    Parses the log file to extract epoch, accuracy, and loss (cross-entropy) metrics.

    Args:
        file_path (str): Path to the training log file.

    Returns:
        epochs (list): List of epoch numbers.
        accuracies (list): List of training accuracies per epoch.
        cross_entropies (list): List of training losses (cross-entropy) per epoch.
    """
    epochs = []
    accuracies = []
    cross_entropies = []

    with open(file_path, 'r') as file:
        for line in file:
            # Match epoch-level metrics
            epoch_match = re.search(r"Epoch (\d+)/\d+: Train Loss = ([0-9\.]+), Train Accuracy = ([0-9\.]+)", line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
                cross_entropies.append(float(epoch_match.group(2)))
                accuracies.append(float(epoch_match.group(3)))

    return epochs, accuracies, cross_entropies


log_file_path_age = '/pretrained/logs/train/train_age.log'
log_file_path_gender = '/pretrained/logs/train/train_gender.log'

output_image_path_age = 'training_resnet_age.png'
output_image_path_gender = 'training_resnet_gender.png'

epochs_age, accuracies_age, cross_entropies_age = parse_log_file(log_file_path_age)
epochs_gender, accuracies_gender, cross_entropies_gender = parse_log_file(log_file_path_gender)

plot_metrics(epochs_age, accuracies_age, cross_entropies_age, output_image_path_age)
plot_metrics(epochs_gender, accuracies_gender, cross_entropies_gender, output_image_path_gender)

