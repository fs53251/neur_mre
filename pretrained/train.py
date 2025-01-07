import argparse
import logging
import os
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 for gender or age classification.")
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Path to save model checkpoints')
    parser.add_argument('-t', '--task', type=str, choices=['gender', 'age'], required=True, help='Task: gender or age')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    return parser.parse_args()



# Prepare the dataset
def get_dataloader(rec_path, batch_size, shuffle):
    data_iter = mx.io.ImageRecordIter(
        path_imgrec=rec_path,
        data_shape=(3, 224, 224),  # ResNet input size
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_iter


def train(task, checkpoint_path, use_gpu, num_epochs=20, batch_size=32, lr=0.001):
    # Set paths based on the task
    if task == 'gender':
        train_rec = 'D:/fer/9.sem/NNETS/neur_mre/adience/rec/gender_train.rec'
        val_rec = 'D:/fer/9.sem/NNETS/neur_mre/adience/rec/gender_val.rec'
        num_classes = 2
        log_file = 'logs/train/train_gender.log'
    elif task == 'age':
        train_rec = 'D:/fer/9.sem/NNETS/neur_mre/adience/rec/age_train.rec'
        val_rec = 'D:/fer/9.sem/NNETS/neur_mre/adience/rec/age_val.rec'
        num_classes = 8
        log_file = 'logs/train/train_age.log'

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
    model = models.resnet50_v2(pretrained=True)
    with model.name_scope():
        model.output = nn.Dense(num_classes)
    model.output.initialize(mx.init.Xavier(), ctx=ctx)
    model.collect_params().reset_ctx(ctx)
    model.hybridize()

    # Loss and optimizer
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': 0.9, 'wd': 0.0001})

    # Data loaders
    train_data = get_dataloader(train_rec, batch_size, shuffle=True)
    val_data = get_dataloader(val_rec, batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = 0
        train_samples = 0
        train_correct = 0

        # Training phase
        for batch_id, batch in enumerate(train_data):
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)

            with autograd.record():
                outputs = model(data)
                loss = loss_fn(outputs, label)

            loss.backward()
            trainer.step(data.shape[0])

            # Calculate batch accuracy
            predictions = outputs.argmax(axis=1)
            correct = (predictions == label).sum().asscalar()

            train_loss += loss.sum().asscalar()
            train_samples += data.shape[0]
            train_correct += correct

            # Calculate accuracy
            train_accuracy = train_correct / train_samples

            # Log every 10th batch for progress tracking
            if batch_id % 10 == 0:
                logger.info(f"Batch {batch_id}: "
                            f"Training Loss: {train_loss / (train_samples + 1e-6):.4f}, "
                            f"Training Accuracy: {train_accuracy:.4f}")

        train_loss /= train_samples

        # Validation phase
        val_loss = 0
        correct = 0
        total = 0
        for batch in val_data:
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)
            outputs = model(data)
            loss = loss_fn(outputs, label)
            val_loss += loss.sum().asscalar()
            predictions = outputs.argmax(axis=1)
            correct += (predictions == label).sum().asscalar()
            total += data.shape[0]

        val_loss /= total
        val_accuracy = correct / total

        # Log results at the end of each epoch
        logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Train Accuracy = {train_accuracy:.4f}, "
                    f"Validation Loss = {val_loss:.4f}, "
                    f"Validation Accuracy = {val_accuracy:.4f}")

        # Save checkpoint
        checkpoint_file = os.path.join(checkpoint_path, f'{task}_epoch_{epoch + 1}.params')
        model.save_parameters(checkpoint_file)
        logger.info(f"Epoch {epoch + 1}: Checkpoint saved to {checkpoint_file}")

        # Reset the iterators at the end of each epoch
        train_data.reset()
        val_data.reset()



if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.checkpoint, exist_ok=True)
    train(task=args.task, checkpoint_path=args.checkpoint, use_gpu=args.gpu, num_epochs=args.epochs)

