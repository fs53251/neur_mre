import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models


def parse_args():
    parser = argparse.ArgumentParser(description="Plot confusion matrix for gender classification.")
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model .params file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('-o', '--output', type=str, default='gender_confusion_matrix.png', help='Output file for the plot')
    return parser.parse_args()


# Prepare the dataset
def get_dataloader(rec_path, batch_size):
    data_iter = mx.io.ImageRecordIter(
        path_imgrec=rec_path,
        data_shape=(3, 224, 224),  # ResNet input size
        batch_size=batch_size,
        shuffle=False
    )
    return data_iter


def plot_confusion_matrix(model_path, use_gpu, output_file, batch_size=32):
    test_rec = 'D:/fer/9.sem/NNETS/neur_mre/adience/rec/gender_test.rec'
    num_classes = 2
    class_names = ['Female', 'Male']

    ctx = mx.gpu() if use_gpu and mx.context.num_gpus() > 0 else mx.cpu()

    # Define the model
    model = models.resnet50_v2(pretrained=False)
    with model.name_scope():
        model.output = nn.Dense(num_classes)

    model.output.initialize(mx.init.Xavier(), ctx=ctx)
    model.load_parameters(model_path, ctx=ctx)
    print(f"Model parameters loaded from {model_path}")
    model.collect_params().reset_ctx(ctx)

    test_data = get_dataloader(test_rec, batch_size)

    all_predictions = []
    all_labels = []

    for batch in test_data:
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)

        outputs = model(data)
        predictions = outputs.argmax(axis=1)

        all_predictions.extend(predictions.asnumpy().tolist())
        all_labels.extend(label.asnumpy().tolist())

    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix for Gender Classification")

    # Save the plot
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {output_file}")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    plot_confusion_matrix(model_path=args.model, use_gpu=args.gpu, output_file=args.output)
