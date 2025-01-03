import re
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    """Parses the log file to extract epoch, accuracy, and cross-entropy data."""
    epochs = []
    accuracies = []
    cross_entropies = []

    with open(file_path, 'r') as file:
        for line in file:
            print(f"Processing Line: {line.strip()}")
            
            # Match lines with accuracy and cross-entropy
            accuracy_match = re.search(r"accuracy=([0-9\.]+)", line)
            cross_entropy_match = re.search(r"cross-entropy=([0-9\.]+)", line)
            epoch_match = re.search(r"Epoch\[(\d+)]", line)

            if accuracy_match and cross_entropy_match and epoch_match:
                epochs.append(int(epoch_match.group(1)))
                accuracies.append(float(accuracy_match.group(1)))
                cross_entropies.append(float(cross_entropy_match.group(1)))

    return epochs, accuracies, cross_entropies



def plot_metrics(epochs, accuracies, cross_entropies, output_path):
    """Plots accuracy and cross-entropy metrics and saves the plot to a file."""
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, accuracies, label='Accuracy', marker='x', color='blue')
    plt.plot(epochs, cross_entropies, label='Cross-Entropy', marker='x', color='red')

    plt.title('Training Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

def main():
    log_file_path = '/home/filip/Documents/fer/neur_mre/projekt/logs/train_age_logs/training_0.log'  # Path to the log file
    output_image_path = 'training_metrics_age.png'  # Path to save the plot

    epochs, accuracies, cross_entropies = parse_log_file(log_file_path)

    print(epochs)
    if not epochs:
        print("No valid data found in the log file.")
        return

    plot_metrics(epochs, accuracies, cross_entropies, output_image_path)

if __name__ == "__main__":
    main()
