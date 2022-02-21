import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from due import dkl
from due.wide_resnet import WideResNet
from due.sngp import Laplace

from lib.datasets import get_dataset
from lib.evaluate_ood import get_ood_metrics
from lib.utils import get_results_directory, Hyperparameters, set_seed
from mnist_resnet import MNISTResNet
from params import NNParams

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)

def main(hparams):
    """
    Main function
    """
    results_dir = get_results_directory(hparams.output_dir)
    writer = SummaryWriter(log_dir=str(results_dir))

    ds = get_dataset(hparams.dataset, root=hparams.data_root)
    input_size, num_classes, train_dataset, test_dataset = ds

    hparams.seed = set_seed(hparams.seed)

    if hparams.n_inducing_points is None:
        hparams.n_inducing_points = num_classes

    print(f"Training with {hparams}")
    hparams.save(results_dir / "hparams.json")

    nn_params = NNParams(
            spectral_normalization=False,
            dropout_rate=0.0,
            coeff=0.9,
            n_power_iterations=1,
            batchnorm_momentum=0.01,
            weight_decay=5e-4,
    )
    feature_extractor = MNISTResNet(nn_params)

    
    initial_inducing_points, initial_lengthscale = dkl.initial_values(
        train_dataset, feature_extractor, hparams.n_inducing_points
    )

    gp = dkl.GP(
        num_outputs=num_classes,
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=hparams.kernel,
    )

    model = dkl.DKL(feature_extractor, gp)

    likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)
    likelihood = likelihood.cuda()

    elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_dataset))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    # model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hparams.learning_rate,
        momentum=0.9,
        weight_decay=hparams.weight_decay,
    )

    milestones = [60, 120, 160]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    def step(engine, batch):
        model.train()
        if not hparams.sngp:
            likelihood.train()

        optimizer.zero_grad()

        x, y = batch
        # x, y = x.cuda(), y.cuda()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        if not hparams.sngp:
            likelihood.eval()

        x, y = batch
        # x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            y_pred = model(x)

        return y_pred, y

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average()
    metric.attach(trainer, "loss")

    def output_transform(output):
        y_pred, y = output

        # Sample softmax values independently for classification at test time
        y_pred = y_pred.to_data_independent_dist()

        # The mean here is over likelihood samples
        y_pred = likelihood(y_pred).probs.mean(0)

        return y_pred, y

    if hparams.sngp:
        output_transform = lambda x: x  # noqa

    metric = Accuracy(output_transform=output_transform)
    metric.attach(evaluator, "accuracy")

    if hparams.sngp:
        metric = Loss(F.cross_entropy)
    else:
        metric = Loss(lambda y_pred, y: -likelihood.expected_log_prob(y, y_pred).mean())

    metric.attach(evaluator, "loss")

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_dataset, test_dataset = repeated_mnist.create_repeated_MNIST_dataset(num_repetitions=1, add_noise=False)

    num_initial_samples = 20
    num_classes = 10

    initial_samples = active_learning.get_balanced_sample_indices(
        repeated_mnist.get_targets(train_dataset), num_classes=num_classes, n_per_digit=num_initial_samples / num_classes
    )

    # experiment
    max_training_samples = 150
    acquisition_batch_size = 5
    num_inference_samples = 100
    num_test_inference_samples = 5
    num_samples = 100000

    test_batch_size = 512
    batch_size = 64
    scoring_batch_size = 128
    training_iterations = 4096

    use_cuda = torch.cuda.is_available()

    print(f"use_cuda: {use_cuda}")

    device = "cuda" if use_cuda else "cpu"

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    active_learning_data = active_learning.ActiveLearningData(train_dataset)

    # Split off the initial samples first.
    active_learning_data.acquire(initial_samples)

    # THIS REMOVES MOST OF THE POOL DATA. UNCOMMENT THIS TO TAKE ALL UNLABELLED DATA INTO ACCOUNT!
    # active_learning_data.extract_dataset_from_pool(40000)

    train_loader = torch.utils.data.DataLoader(
        active_learning_data.training_dataset,
        sampler=active_learning.RandomFixedLengthSampler(active_learning_data.training_dataset, training_iterations),
        batch_size=batch_size,
        **kwargs,
    )

    pool_loader = torch.utils.data.DataLoader(
        active_learning_data.pool_dataset, batch_size=scoring_batch_size, shuffle=False, **kwargs
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=hparams.batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     **kwargs,
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=512, shuffle=False, **kwargs
    # )

    if hparams.sngp:

        @trainer.on(Events.EPOCH_STARTED)
        def reset_precision_matrix(trainer):
            model.reset_precision_matrix()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        train_loss = metrics["loss"]

        result = f"Train - Epoch: {trainer.state.epoch} "
        if hparams.sngp:
            result += f"Loss: {train_loss:.2f} "
        else:
            result += f"ELBO: {train_loss:.2f} "
        print(result)

        writer.add_scalar("Loss/train", train_loss, trainer.state.epoch)

        if hparams.spectral_conv:
            for name, layer in model.feature_extractor.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    writer.add_scalar(
                        f"sigma/{name}", layer.weight_sigma, trainer.state.epoch
                    )

        if trainer.state.epoch > 150 and trainer.state.epoch % 5 == 0:
            _, auroc, aupr = get_ood_metrics(
                hparams.dataset, "SVHN", model, likelihood, hparams.data_root
            )
            print(f"OoD Metrics - AUROC: {auroc}, AUPR: {aupr}")
            writer.add_scalar("OoD/auroc", auroc, trainer.state.epoch)
            writer.add_scalar("OoD/auprc", aupr, trainer.state.epoch)

        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        test_loss = metrics["loss"]

        result = f"Test - Epoch: {trainer.state.epoch} "
        if hparams.sngp:
            result += f"Loss: {test_loss:.2f} "
        else:
            result += f"NLL: {test_loss:.2f} "
        result += f"Acc: {acc:.4f} "
        print(result)
        writer.add_scalar("Loss/test", test_loss, trainer.state.epoch)
        writer.add_scalar("Accuracy/test", acc, trainer.state.epoch)

        scheduler.step()

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    trainer.run(train_loader, max_epochs=10)

    # Done training - time to evaluate
    results = {}

    evaluator.run(test_loader)
    test_acc = evaluator.state.metrics["accuracy"]
    test_loss = evaluator.state.metrics["loss"]
    results["test_accuracy"] = test_acc
    results["test_loss"] = test_loss

    _, auroc, aupr = get_ood_metrics(
        hparams.dataset, "SVHN", model, likelihood, hparams.data_root
    )
    results["auroc_ood_svhn"] = auroc
    results["aupr_ood_svhn"] = aupr

    print(f"Final accuracy {results['test_accuracy']:.4f}")

    results_json = json.dumps(results, indent=4, sort_keys=True)
    (results_dir / "results.json").write_text(results_json)

    torch.save(model.state_dict(), results_dir / "model.pt")
    if likelihood is not None:
        torch.save(likelihood.state_dict(), results_dir / "likelihood.pt")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to use for training"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate",
    )

    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")

    parser.add_argument(
        "--dataset",
        default="CIFAR10",
        choices=["CIFAR10", "CIFAR100", "MNIST"],
        help="Pick a dataset",
    )

    parser.add_argument(
        "--kernel",
        default="RBF",
        choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],
        help="Pick a kernel",
    )

    parser.add_argument(
        "--no_spectral_conv",
        action="store_false",
        dest="spectral_conv",
        help="Don't use spectral normalization on the convolutions",
    )

    parser.add_argument(
        "--no_spectral_bn",
        action="store_false",
        dest="spectral_bn",
        help="Don't use spectral normalization on the batch normalization layers",
    )

    parser.add_argument(
        "--sngp",
        action="store_true",
        help="Use SNGP (RFF and Laplace) instead of a DUE (sparse GP)",
    )

    parser.add_argument(
        "--n_inducing_points", type=int, help="Number of inducing points"
    )

    parser.add_argument("--seed", type=int, help="Seed to use for training")

    parser.add_argument(
        "--coeff", type=float, default=3, help="Spectral normalization coefficient"
    )

    parser.add_argument(
        "--n_power_iterations", default=1, type=int, help="Number of power iterations"
    )

    parser.add_argument(
        "--output_dir", default="./default", type=str, help="Specify output directory"
    )
    parser.add_argument(
        "--data_root", default="./data", type=str, help="Specify data directory"
    )

    args = parser.parse_args()
    hparams = Hyperparameters(**vars(args))

    main(hparams)