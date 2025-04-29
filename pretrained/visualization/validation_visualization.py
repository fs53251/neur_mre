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
    val_accuracies = []
    val_losses = []

    with open(file_path, 'r') as file:
        for line in file:
            val_match = re.search(r"Epoch (\d+)/\d+: .*Validation Loss = ([0-9\.]+), Validation Accuracy = ([0-9\.]+)",
                                  line)
            if val_match:
                epochs.append(int(val_match.group(1)))
                val_losses.append(float(val_match.group(2)))
                val_accuracies.append(float(val_match.group(3)))

    return epochs, val_accuracies, val_losses


log_file_path_age = 'D:/fer/9.sem/NNETS/neur_mre/pretrained/logs/train/train_age.log'
log_file_path_gender = 'D:/fer/9.sem/NNETS/neur_mre/pretrained/logs/train/train_gender.log'

output_image_path_age = 'validation_resnet_age.png'
output_image_path_gender = 'validation_resnet_gender.png'

epochs_age, accuracies_age, cross_entropies_age = parse_log_file(log_file_path_age)
epochs_gender, accuracies_gender, cross_entropies_gender = parse_log_file(log_file_path_gender)

plot_metrics(epochs_age, accuracies_age, cross_entropies_age, output_image_path_age)
plot_metrics(epochs_gender, accuracies_gender, cross_entropies_gender, output_image_path_gender)

