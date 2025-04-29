import argparse
import logging
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the ResNet-50 model for gender or age classification.")
    parser.add_argument('-t', '--task', type=str, choices=['gender', 'age'], required=True, help='Task: gender or age')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model .params file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
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


# Define the testing function
def test(task, model_path, use_gpu, batch_size=32):
    # Set paths based on the task
    if task == 'gender':
        test_rec = 'D:/fer/9.sem/NNETS/neur_mre/adience/rec/gender_test.rec'
        num_classes = 2
        log_file = 'logs/test/test_gender.log'
        calculate_one_off = False
    elif task == 'age':
        test_rec = 'D:/fer/9.sem/NNETS/neur_mre/adience/rec/age_test.rec'
        num_classes = 8
        log_file = 'logs/test/test_age.log'
        calculate_one_off = True  # One-off accuracy is calculated only for 'age' task

    # Logging setup
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logger = logging.getLogger()

    # Set the context (GPU if available, otherwise CPU)
    ctx = mx.gpu() if use_gpu and mx.context.num_gpus() > 0 else mx.cpu()

    # Define the model
    model = models.resnet50_v2(pretrained=False)
    with model.name_scope():
        model.output = nn.Dense(num_classes)

    # Initialize the model's parameters before resetting the context
    model.output.initialize(mx.init.Xavier(), ctx=ctx)  # Ensure output layer is initialized

    # Load the model parameters
    model.load_parameters(model_path, ctx=ctx)
    logger.info(f"Model parameters loaded from {model_path}")

    model.collect_params().reset_ctx(ctx)  # Reset context for all parameters

    # Loss function
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    # Data loader
    test_data = get_dataloader(test_rec, batch_size)

    # Testing loop
    correct = 0
    total = 0
    test_loss = 0
    one_off_correct = 0  # Counter for one-off accuracy
    one_off_total = 0  # Total samples for one-off accuracy

    for batch in test_data:
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)

        outputs = model(data)
        loss = loss_fn(outputs, label)

        test_loss += loss.sum().asscalar()

        # Calculate accuracy
        predictions = outputs.argmax(axis=1)
        correct += (predictions == label).sum().asscalar()
        total += data.shape[0]

        if calculate_one_off:
            # Calculate one-off accuracy (predicted label within 1 of true label)
            one_off_correct += ((mx.nd.abs(predictions - label) <= 1).sum()).asscalar()
            one_off_total += data.shape[0]

    test_loss /= total
    test_accuracy = correct / total

    # If task is age, calculate one-off accuracy
    if calculate_one_off:
        one_off_accuracy = one_off_correct / one_off_total
        logger.info(f"One-Off Accuracy = {one_off_accuracy:.4f}")
        print(f"One-Off Accuracy = {one_off_accuracy:.4f}")

    # Log the test results
    logger.info(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")


if __name__ == '__main__':
    args = parse_args()
    test(task=args.task, model_path=args.model, use_gpu=args.gpu)
