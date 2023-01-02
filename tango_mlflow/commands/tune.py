import argparse
import copy
import dataclasses
import json
import os
import tempfile
from functools import partial
from typing import List, Optional

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from tango.__main__ import _run as tango_run
from tango.common import FromParams, Params
from tango.settings import TangoGlobalSettings
from tango.workspace import Workspace

from tango_mlflow.commands.subcommand import Subcommand
from tango_mlflow.util import RunKind, generate_unique_run_name, get_mlflow_run_by_tango_run
from tango_mlflow.workspace import MLFlowWorkspace


@Subcommand.register("tune")
class TuneCommand(Subcommand):
    """Tune hyper-parameters using Optuna."""

    def setup(self) -> None:
        self.parser.add_argument(
            "experiment",
            type=str,
        )
        self.parser.add_argument(
            "hparam_path",
            type=str,
            help="path to hyperparameter file",
        )
        self.parser.add_argument(
            "--name",
            type=str,
            help="name of tango run",
        )
        self.parser.add_argument(
            "--tango-settings",
            type=str,
            help="path to tango settings file",
        )
        self.parser.add_argument(
            "--metric",
            type=str,
            help="The metric you want to optimize.",
            default="best_validation_loss",
        )
        self.parser.add_argument(
            "--num-trials",
            type=int,
            help="The number of trials. If this argument is not given, as many trials run as possible.",
        )
        self.parser.add_argument(
            "--direction",
            type=str,
            choices=("minimize", "maximize"),
            help="Set direction of optimization to a new study. Set 'minimize' "
            "for minimization and 'maximize' for maximization.",
        )
        self.parser.add_argument(
            "--timeout",
            type=float,
            help="Stop study after the given number of second(s). If this argument"
            " is not given, as many trials run as possible.",
        )
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume optimization if a study with the same name already exists.",
        )
        self.parser.add_argument(
            "--optuna-settings",
            type=str,
            help="path to optuna settings file",
        )

    def run(self, args: argparse.Namespace) -> None:
        import optuna
        from optuna.pruners import BasePruner
        from optuna.samplers import BaseSampler

        @dataclasses.dataclass
        class OptunaSettings(FromParams):
            """Settings for Optuna."""

            n_trials: Optional[int] = None
            metric: Optional[str] = None
            direction: str = "minimize"
            sampler: Optional[BaseSampler] = None
            pruner: Optional[BasePruner] = None
            timeout: int = 600

        def _objective(
            trial: optuna.trial.Trial,
            experiment: str,
            hparam_path: str,
            tango_settings: TangoGlobalSettings,
        ) -> float:
            tango_settings = copy.deepcopy(tango_settings)

            ext_var: List[str] = []
            with open(hparam_path) as jsonfile:
                for key, params in json.load(jsonfile).items():
                    value_type = params.pop("type")
                    suggest = getattr(trial, f"suggest_{value_type}")
                    value = suggest(key, **params)
                    ext_var.append(f"{key}={value}")

            run_name = tango_run(
                settings=tango_settings,
                experiment=experiment,
                name=f"{trial.study.study_name}_{trial.number}",
                ext_var=ext_var,
            )

            workspace = Workspace.from_params(Params(tango_settings.workspace or {}))
            assert isinstance(workspace, MLFlowWorkspace)
            mlflow_run = get_mlflow_run_by_tango_run(
                workspace.mlflow_client,
                workspace.experiment_name,
                tango_run=run_name,
            )
            if mlflow_run is None:
                raise RuntimeError(f"Could not find MLFlow run for tango run {run_name}")

            metric_history = workspace.mlflow_client.get_metric_history(mlflow_run.info.run_id, args.metric)
            if not metric_history:
                raise RuntimeError(f"No metric history found for {args.metric}.")

            latest_metric = metric_history[-1]

            return float(latest_metric.value)

        tango_settings = (
            TangoGlobalSettings.from_file(args.tango_settings) if args.tango_settings else TangoGlobalSettings.default()
        )
        workspace = Workspace.from_params(tango_settings.workspace or {})
        if not isinstance(workspace, MLFlowWorkspace):
            raise ValueError("Tango workspace type must be mlflow.")

        optuna_settings = (
            OptunaSettings.from_params(Params.from_file(args.optuna_settings), **vars(args))
            if args.optuna_settings
            else OptunaSettings.from_params(Params(vars(args)))
        )
        study_name = args.name or generate_unique_run_name(workspace.mlflow_client, workspace.experiment_name)
        optuna_storage_path = tempfile.mktemp()

        mlflow_run = get_mlflow_run_by_tango_run(
            workspace.mlflow_client,
            workspace.experiment_name,
            tango_run=study_name,
        )
        if mlflow_run is not None:
            if not args.resume:
                raise ValueError(f"Run {args.name} already exists. If you want to resume, use --resume option.")
            mlflow.artifacts.download_artifacts(
                run_id=mlflow_run.info.run_id,
                artifact_path="optuna.db",
                dst_path=optuna_storage_path,
            )

        study = optuna.create_study(
            study_name=study_name,
            direction=optuna_settings.direction,
            sampler=optuna_settings.sampler,
            pruner=optuna_settings.pruner,
            storage=f"sqlite:///{optuna_storage_path}",
            load_if_exists=args.resume,
        )

        with mlflow.start_run(
            run_id=mlflow_run.info.run_id if mlflow_run is not None else None,
            run_name=study_name if mlflow_run is None else None,
            tags={"job_type": RunKind.OPTUNA_STUDY.value},
        ) as active_run:
            workspace._mlflow_tags[MLFLOW_PARENT_RUN_ID] = active_run.info.run_id
            tango_settings.workspace = {"type": "from_url", "url": workspace.url}
            objective = partial(
                _objective,
                hparam_path=args.hparam_path,
                tango_settings=tango_settings,
            )

            mlflow.log_params(tango_settings.to_params().as_flat_dict())
            mlflow.log_artifact(args.experiment, "experiment.jsonnet")
            mlflow.log_artifact(args.hparam_path, "hparams.json")

            try:
                study.optimize(
                    objective,
                    n_trials=optuna_settings.n_trials,
                    timeout=optuna_settings.timeout,
                )
                mlflow.log_metric("best_metric", study.best_trial.value)
                mlflow.log_metric("best_trial", study.best_trial.number)
                with tempfile.TemporaryDirectory() as tempdir:
                    filename = os.path.join(tempdir, "best_params.json")
                    with open(filename, "w") as jsonfile:
                        json.dump(study.best_trial.params, jsonfile)
                    mlflow.log_artifact(filename, "best_params.json")
            finally:
                mlflow.log_artifact(optuna_storage_path, "optuna.db")
                os.remove(optuna_storage_path)
