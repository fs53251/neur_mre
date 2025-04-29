import re

def parse_log_file(file_path):
    """
    Parses the log file to extract epoch, training, and validation metrics.

    Args:
        file_path (str): Path to the training log file.

    Returns:
        epochs (list): List of epoch numbers.
        train_losses (list): List of training losses per epoch.
        train_accuracies (list): List of training accuracies per epoch.
        val_losses (list): List of validation losses per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
    """
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    with open(file_path, 'r') as file:
        for line in file:
            # Match metrics for both training and validation
            match = re.search(
                r"Epoch (\d+)/\d+: Train Loss = ([0-9\.]+), Train Accuracy = ([0-9\.]+), "
                r"Validation Loss = ([0-9\.]+), Validation Accuracy = ([0-9\.]+)",
                line
            )
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accuracies.append(float(match.group(3)))
                val_losses.append(float(match.group(4)))
                val_accuracies.append(float(match.group(5)))

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies


def plot_combined_metrics(
    epochs, train_losses, train_accuracies, val_losses, val_accuracies, output_image_path
):
    """
    Plots combined training and validation metrics (loss and accuracy).

    Args:
        epochs (list): List of epoch numbers.
        train_losses (list): List of training losses per epoch.
        train_accuracies (list): List of training accuracies per epoch.
        val_losses (list): List of validation losses per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
        output_image_path (str): Path to save the plot image.
    """
    import matplotlib.pyplot as plt

    # Create subplots for loss and accuracy
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot losses
    ax1.plot(epochs, train_losses, label="Train Loss", color="green")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Plot accuracies
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accuracies, label="Train Accuracy", color="green", linestyle="--")
    ax2.plot(epochs, val_accuracies, label="Validation Accuracy", color="blue", linestyle="--")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")

    plt.title("Training and Validation Metrics")
    plt.savefig(output_image_path)
    plt.close()


# File paths
log_file_path_age = "D:/fer/9.sem/NNETS/neur_mre/pretrained/logs/train/train_age.log"
log_file_path_gender = "D:/fer/9.sem/NNETS/neur_mre/pretrained/logs/train/train_gender.log"

output_image_path_age = "combined_metrics_age.png"
output_image_path_gender = "combined_metrics_gender.png"

# Parse log files
epochs_age, train_losses_age, train_accuracies_age, val_losses_age, val_accuracies_age = parse_log_file(log_file_path_age)
epochs_gender, train_losses_gender, train_accuracies_gender, val_losses_gender, val_accuracies_gender = parse_log_file(log_file_path_gender)

# Plot and save metrics
plot_combined_metrics(epochs_age, train_losses_age, train_accuracies_age, val_losses_age, val_accuracies_age, output_image_path_age)
plot_combined_metrics(epochs_gender, train_losses_gender, train_accuracies_gender, val_losses_gender, val_accuracies_gender, output_image_path_gender)
