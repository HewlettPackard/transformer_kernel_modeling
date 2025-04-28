import os

import click
import gpytorch
import numpy as np
import pandas as pd
import torch
import tqdm
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_standardized_log_loss,
    negative_log_predictive_density,
)
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, ard_num_dims=inducing_points.shape[1])
            # + SpectralMixtureKernel(num_mixtures = 3, ard_num_dims=inducing_points.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_varmodel_from_df(data_df,
                           kernel_name,
                           gpu_name,
                           target_dir,
                           batch_size=256,
                           epochs=50,
                           lr=0.01,
                           inducing_points=200,
                           test_split=1/5):
    df = data_df.reset_index()
    df = df.drop(
        columns=[
            "gpu_name",
            "kernel",
            "phase",
            "sms",
            "tpcs",
            "gpcs",
            "cuda_cores",
            "tensor_cores",
            "ram_gb",
            "l2_cache_mb",
            "mem_bw_gB_s",
            "index",
        ]
    )

    df = df.dropna()

    print(f"kernel: {kernel_name}, gpu: {gpu_name}, df.shape: {df.shape}")

    # df_scaler = preprocessing.StandardScaler().fit(df)
    # df_scaler.set_output(transform="pandas")
    # df = df_scaler.transform(df)

    # with open(f"{target_dir}/{kernel_name}_standardscaler.pkl", "wb") as target_file:
    #     pickle.dump(df_scaler, target_file)

    df_x = df.drop(columns=["runtime_s"]).to_numpy()
    df_y = df[["runtime_s"]].to_numpy()

    if test_split == -1:
        # No test split, use all data for both training and testing
        train_x, train_y = df_x, df_y
        test_x, test_y = train_x, train_y
        print("Using all data for both training and testing (no split)")
    else:
        # Normal train-test split
        train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=test_split)

    lin_reg = LinearRegression().fit(train_x, np.log(train_y + 1e-9))
    lin_reg_score = lin_reg.score(test_x, np.log(test_y + 1e-9))
    lin_reg_rmse = np.sqrt(np.mean((np.exp(lin_reg.predict(test_x)) - test_y) ** 2))

    train_x = torch.tensor(train_x).contiguous()
    train_y = torch.tensor(train_y).flatten().contiguous()
    test_x = torch.tensor(test_x).contiguous()
    test_y = torch.tensor(test_y).flatten().contiguous()

    train_x = train_x.to(torch.float64)
    # Take the log of the target variable
    train_y = torch.log(train_y.to(torch.float64) + 1e-9)
    test_x = test_x.to(torch.float64)
    # Take the log of the target variable
    test_y = torch.log(test_y.to(torch.float64) + 1e-9)

    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

    batch_size = batch_size

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # initialize likelihood and model
    # inducing_points = torch.randn(3000, train_x.size(1))
    # inducing_points = train_x[:int(0.1 * train_x.size(0)), :]
    inducing_points = train_x[:inducing_points, :]

    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = model.double()
    likelihood = likelihood.double()

    model = model.cuda()
    likelihood = likelihood.cuda()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    epochs = epochs
    epochs_iter = tqdm.tqdm(range(epochs), desc="Epoch")

    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # means = torch.tensor([0.0])
    # lowers = torch.tensor([0.0])
    # uppers = torch.tensor([0.0])

    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        lower, upper = observed_pred.confidence_region()

    means = observed_pred.mean.cpu()
    lowers = lower.cpu()
    uppers = upper.cpu()

    # with torch.no_grad():
    #     for x_batch, y_batch in test_loader:
    #         preds = likelihood(model(x_batch))
    #         _lowers, _uppers = preds.confidence_region()

    #         means = torch.cat([means, preds.mean.cpu()])
    #         lowers = torch.cat([lowers, _lowers.cpu()])
    #         uppers = torch.cat([uppers, _uppers.cpu()])

    # means = means[1:]
    # lowers = lowers[1:]
    # uppers = uppers[1:]

    test_y = test_y.cpu()

    evaluation = {
        # "scaler": df_scaler,
        "mean_test_y": torch.mean(torch.exp(test_y)),
        "min_test_y": torch.min(torch.exp(test_y)),
        "max_test_y": torch.max(torch.exp(test_y)),
        "NLPD": negative_log_predictive_density(observed_pred, test_y.cuda()),
        "MLSS": mean_standardized_log_loss(observed_pred, test_y.cuda()),
        "MAE": mean_absolute_error(observed_pred, test_y.cuda()),
        "RMSE": torch.sqrt(
            torch.mean((means.exp() - test_y.exp()) ** 2),
        ),
        "RMSE_gpytorch": torch.sqrt(
            mean_squared_error(observed_pred, test_y.cuda(), squared=True)
        ),
        "R2": r2_score(means.exp(), test_y.exp()),
        "lin_reg_R2": lin_reg_score,
        "lin_reg_RMSE": lin_reg_rmse,
        "out_of_confidence": torch.div(
            torch.sum(torch.logical_or(lowers > test_y, uppers < test_y)),
            test_y.size(dim=0),
        ),
    }

    torch.save(model.state_dict(), f"{target_dir}/{kernel_name}_model_state.pth")
    torch.save(
        likelihood.state_dict(), f"{target_dir}/{kernel_name}_likelihood_state.pth"
    )

    del train_x
    del train_y
    del test_x
    del test_y

    del model
    del likelihood
    del means
    del uppers
    del lowers

    # return model, likelihood, evaluation
    return evaluation


def train_generalist_varmodel_from_df(data_df, target_dir, test_split = 0.2):
    df = data_df.reset_index()
    df = df.drop(
        columns=[
            "phase",
            "index",
        ]
    )

    df = df.dropna()

    df_x = df.drop(columns=["runtime_s"]).to_numpy()
    df_y = df[["runtime_s"]].to_numpy()

    if test_split == -1:
        # No test split, use all data for both training and testing
        train_x, train_y = df_x, df_y
        test_x, test_y = train_x, train_y
        print("Using all data for both training and testing (no split)")
    else:
        # Normal train-test split
        train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=test_split)

    lin_reg = LinearRegression().fit(train_x, train_y)
    lin_reg_score = lin_reg.score(test_x, test_y)
    lin_reg_rmse = np.sqrt(np.mean((lin_reg.predict(test_x) - test_y) ** 2))

    train_x = torch.tensor(train_x).contiguous()
    train_y = torch.tensor(train_y).flatten().contiguous()
    test_x = torch.tensor(test_x).contiguous()
    test_y = torch.tensor(test_y).flatten().contiguous()

    train_x = train_x.to(torch.float64)
    train_y = train_y.to(torch.float64)
    test_x = test_x.to(torch.float64)
    test_y = test_y.to(torch.float64)

    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

    batch_size = 1024

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    inducing_points = train_x[:2000, :]

    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = model.double()
    likelihood = likelihood.double()

    model = model.cuda()
    likelihood = likelihood.cuda()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.01,
    )

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    epochs = 50
    epochs_iter = tqdm.tqdm(range(epochs), desc="Epoch")

    for i in epochs_iter:
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        lower, upper = observed_pred.confidence_region()

    means = observed_pred.mean.cpu()
    lowers = lower.cpu()
    uppers = upper.cpu()

    test_y = test_y.cpu()

    evaluation = {
        "mean_test_y": torch.mean(test_y),
        "min_test_y": torch.min(test_y),
        "max_test_y": torch.max(test_y),
        "NLPD": negative_log_predictive_density(observed_pred, test_y.cuda()),
        "MLSS": mean_standardized_log_loss(observed_pred, test_y.cuda()),
        "MAE": mean_absolute_error(observed_pred, test_y.cuda()),
        "RMSE": torch.sqrt(
            mean_squared_error(observed_pred, test_y.cuda(), squared=True)
        ),
        "R2": r2_score(means, test_y),
        "lin_reg_R2": lin_reg_score,
        "lin_reg_RMSE": lin_reg_rmse,
        "out_of_confidence": torch.div(
            torch.sum(torch.logical_or(lowers > test_y, uppers < test_y)),
            test_y.size(dim=0),
        ),
    }

    torch.save(model.state_dict(), f"{target_dir}/generalist_model_state.pth")
    torch.save(
        likelihood.state_dict(), f"{target_dir}/generalist_likelihood_state.pth"
    )

    del train_x
    del train_y
    del test_x
    del test_y

    del model
    del likelihood
    del means
    del uppers
    del lowers

    return evaluation


@click.command()
@click.option('--storage-root-dir', default='results/sc25/model_training_summary', help='Root directory for storage')
@click.option('--results-root-dir', default='results/sc25/raw_kernel_profiling_data', help='Root directory for results')
@click.option('--data-suffix', default='_scaled_experiments.csv', help='Suffix for kernel data files')
@click.option('--summary-filename', default='all_kernels_model_evaluation.csv', help='Filename for the summary dataframe')
@click.option('--generalist', is_flag=True, help='Flag to toggle training on aggregated GPU and kernel data')
@click.option('--batch-size', default=64, help='Batch size for training')
@click.option('--epochs', default=200, help='Number of epochs for training')
@click.option('--lr', default=0.01, help='Learning rate for training')
@click.option('--inducing-points', default=3000, help='Number of inducing points for training')
@click.option('--test-split', default=0.2, help='Test split ratio (use -1 for no test split)')
def train_all_models(storage_root_dir,
                     results_root_dir,
                     data_suffix,
                     summary_filename,
                     generalist,
                     batch_size,
                     epochs,
                     lr,
                     inducing_points,
                     test_split):
    complete_results = {
        # "a100_meantests": ["rms_norm", "res_add", "mlp", "attn"],
        # "a100_mean_factorial_expanded": ["rms_norm", "res_add", "mlp", "attn"],
        # "a100_mean_factorial": ["rms_norm", "res_add", "mlp", "attn"],
        # "a100": ["rms_norm", "res_add", "mlp", "attn"],
        "h100_mean_factorial_expanded": ["rms_norm", "res_add", "mlp", "attn"],
    }

    kernel_results = {}

    for gpu, results in complete_results.items():
        kernel_results[gpu] = {}

        for kernel in results:
            kernel_results[gpu][kernel] = pd.read_csv(
                f"{results_root_dir}/{gpu}/{kernel}{data_suffix}"
            )

    summary_df = pd.DataFrame()

    if generalist:
        all_data = pd.concat(
            [kernel for gpus in kernel_results.values() for kernel in gpus.values()]
        )
        target_dir = f"{storage_root_dir}/generalist"

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        evaluation = train_generalist_varmodel_from_df(all_data, target_dir)

        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    {
                        "name": ["generalist"],
                        "gpu": ["generalist"],
                        "NLPD": [evaluation["NLPD"].item()],
                        "MLSS": [evaluation["MLSS"].item()],
                        "MAE": [evaluation["MAE"].item()],
                        "RMSE": [evaluation["RMSE"].item()],
                        "R2": [evaluation["R2"].item()],
                        "lin_reg_R2": [evaluation["lin_reg_R2"]],
                        "lin_reg_RMSE": [evaluation["lin_reg_RMSE"]],
                        "out_of_confidence": [evaluation["out_of_confidence"].item()],
                        "mean_test_y": [evaluation["mean_test_y"].item()],
                        "min_test_y": [evaluation["min_test_y"].item()],
                        "max_test_y": [evaluation["max_test_y"].item()],
                    }
                ),
            ]
        )
        print(summary_df)

    else:
        for gpu_name, kernels in kernel_results.items():
            for kernel_name, dataframe in kernels.items():
                target_dir = f"{storage_root_dir}/{gpu_name}_specific"

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                evaluation = train_varmodel_from_df(
                    dataframe, kernel_name, gpu_name, target_dir, batch_size, epochs, lr, inducing_points, test_split
                )

                summary_df = pd.concat(
                    [
                        summary_df,
                        pd.DataFrame(
                            {
                                "name": [kernel_name],
                                "gpu": [gpu_name],
                                "NLPD": [evaluation["NLPD"].item()],
                                "MLSS": [evaluation["MLSS"].item()],
                                "MAE": [evaluation["MAE"].item()],
                                "RMSE": [evaluation["RMSE"].item()],
                                "R2": [evaluation["R2"].item()],
                                "lin_reg_R2": [evaluation["lin_reg_R2"]],
                                "lin_reg_RMSE": [evaluation["lin_reg_RMSE"]],
                                "out_of_confidence": [
                                    evaluation["out_of_confidence"].item()
                                ],
                                "mean_test_y": [evaluation["mean_test_y"].item()],
                                "min_test_y": [evaluation["min_test_y"].item()],
                                "max_test_y": [evaluation["max_test_y"].item()],
                            }
                        ),
                    ]
                )

                summary_df.to_csv(
                    f"{target_dir}/{summary_filename}", index=False
                )
                print(summary_df)


def generate_prediction_slice(search_space, dimension="n_head"):
    sliced_scaled_space = {
        key: [len(value) // 2 for i in range(len(search_space[dimension]))]
        if key != dimension
        else [i for i in range(len(value))]
        for key, value in search_space.items()
    }

    sliced_normalized_space = {
        key: [
            (value[0] / (len(search_space[key]) - 1))
            for i in range(len(search_space[dimension]))
        ]
        if key != dimension
        else [i / (len(search_space[key]) - 1) for i in value]
        for key, value in sliced_scaled_space.items()
    }

    return {
        "sliced_normalized_space": sliced_normalized_space,
        "sliced_scaled_space": sliced_scaled_space,
    }


gpu_search_space = pd.DataFrame(
    {
        "gpu_name": ["h100", "a100", "v100"],
        "sms": [114, 108, 80],
        "tpcs": [57, 54, 40],
        "gpcs": [8, 7, 6],
        "cuda_cores": [14592, 6912, 5120],
        "tensor_cores": [456, 432, 640],
        "ram_gb": [80, 40, 32],
        "l2_cache_mb": [50, 40, 6],
        "mem_bw_gB_s": [2000, 1555, 900],
    }
)


kernel_search_spaces = {
    "attn": {
        "batch": [i for i in range(1, 33)],
        "max_seq_length": [2**i for i in range(8, 14)],
        "n_embed": [2**i for i in range(9, 14)],
        "n_head": [2**i for i in range(4, 11)],
        "tensor_parallelism": [1, 2, 4, 8],
    },
    "mlp": {
        "batch": [i for i in range(1, 33)],
        "seq_length": [2**i for i in range(8, 14)],
        "n_embed": [2**i for i in range(9, 14)],
        "n_head": [2**i for i in range(4, 11)],
        "tensor_parallelism": [1, 2, 4, 8],
    },
    "res_add": {
        "batch": [i for i in range(1, 33)],
        "seq_length": [2**i for i in range(8, 14)],
        "n_embed": [2**i for i in range(9, 14)],
    },
    "rms_norm": {
        "batch": [i for i in range(1, 33)],
        "seq_length": [2**i for i in range(8, 14)],
        "n_embed": [2**i for i in range(9, 14)],
    },
}

if __name__ == '__main__':
    train_all_models()
