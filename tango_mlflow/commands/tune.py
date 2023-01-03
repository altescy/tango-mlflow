import argparse
import copy
import dataclasses
import json
import os
import tempfile
from functools import partial
from logging import getLogger
from typing import Any, Dict, List, Optional, cast

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from tango.__main__ import _run as tango_run
from tango.common import FromParams, Params, Registrable
from tango.common.logging import initialize_logging
from tango.common.util import import_extra_module
from tango.settings import TangoGlobalSettings
from tango.workspace import Workspace

from tango_mlflow.commands.subcommand import Subcommand
from tango_mlflow.util import (
    RunKind,
    build_filter_string,
    generate_unique_run_name,
    get_mlflow_run_by_tango_run,
    is_all_child_run_finished,
)
from tango_mlflow.workspace import MLFlowWorkspace

logger = getLogger(__name__)


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
            help="name of optuna study",
        )
        self.parser.add_argument(
            "--tango-settings",
            type=str,
            help="path to tango settings file",
        )
        self.parser.add_argument(
            "--metric",
            type=str,
            help="metric you want to optimize",
        )
        self.parser.add_argument(
            "--n-trials",
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
            default="optuna.yml",
            help="path to optuna settings file",
        )

    def run(self, args: argparse.Namespace) -> None:
        import optuna
        from optuna.pruners import BasePruner
        from optuna.samplers import BaseSampler

        class OptunaPruner(BasePruner, Registrable):  # type: ignore[misc]
            ...

        OptunaPruner.register("median")(optuna.pruners.MedianPruner)
        OptunaPruner.register("nop")(optuna.pruners.NopPruner)
        OptunaPruner.register("percentile")(optuna.pruners.PercentilePruner)
        OptunaPruner.register("patient")(optuna.pruners.PatientPruner)
        OptunaPruner.register("successive_halving")(optuna.pruners.SuccessiveHalvingPruner)
        OptunaPruner.register("threshold")(optuna.pruners.ThresholdPruner)
        OptunaPruner.register("hyperband")(optuna.pruners.HyperbandPruner)

        class OptunaSampler(BaseSampler, Registrable):  # type: ignore[misc]
            ...

        OptunaSampler.register("grid")(optuna.samplers.GridSampler)
        OptunaSampler.register("random")(optuna.samplers.RandomSampler)
        OptunaSampler.register("tpe")(optuna.samplers.TPESampler)
        OptunaSampler.register("cmaes")(optuna.samplers.CmaEsSampler)
        OptunaSampler.register("partial_fixed")(optuna.samplers.PartialFixedSampler)
        OptunaSampler.register("nsga2")(optuna.samplers.NSGAIISampler)
        OptunaSampler.register("motpe")(optuna.samplers.MOTPESampler)
        OptunaSampler.register("qmcsampler")(optuna.samplers.QMCSampler)

        @dataclasses.dataclass
        class OptunaSettings(FromParams):
            """Settings for Optuna."""

            metric: Optional[str] = None
            direction: str = "minimize"
            sampler: Optional[OptunaSampler] = None
            pruner: Optional[OptunaPruner] = None

            _original_params: Optional[Params] = None

            @classmethod
            def from_args(cls, args: argparse.Namespace) -> "OptunaSettings":
                if args.optuna_settings is not None and os.path.exists(args.optuna_settings):
                    logger.info(f"Loading Optuna settings from {args.optuna_settings}")
                    params = Params.from_file(args.optuna_settings)
                else:
                    params = Params({})
                settings = cls.from_params(copy.deepcopy(params), _original_params=params)
                if args.metric is not None:
                    settings.metric = args.metric
                if args.direction is not None:
                    settings.direction = args.direction
                return settings

            def _to_params(self) -> Dict[str, Any]:
                if self._original_params is not None:
                    return cast(Dict[str, Any], self._original_params.as_dict())
                return super()._to_params()

        def get_mlflow_run_by_optuna_study(
            mlflow_client: mlflow.tracking.MlflowClient,
            experiment_name: str,
            study_name: str,
        ) -> Optional[mlflow.entities.Run]:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_name} does not exist.")
            mlflow_runs = mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=build_filter_string(
                    run_kind=RunKind.OPTUNA_STUDY,
                    additional_filter_string=f"tags.{MLFLOW_RUN_NAME} = '{study_name}'",
                ),
            )
            if not mlflow_runs:
                return None
            if len(mlflow_runs) > 1:
                raise ValueError(f"Found more than one run name '{study_name}' in MLflow experiment")
            return mlflow_runs[0]

        def _objective(
            trial: optuna.trial.Trial,
            metric: str,
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
                settings=copy.deepcopy(tango_settings),
                experiment=experiment,
                name=f"{trial.study.study_name}-{trial.number}",
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

            if mlflow_run.info.status == "RUNNING":
                status = (
                    "FINISHED"
                    if is_all_child_run_finished(
                        workspace.mlflow_client,
                        workspace.experiment_name,
                        mlflow_run=mlflow_run,
                    )
                    else "FAILED"
                )
                workspace.mlflow_client.set_terminated(mlflow_run.info.run_id, status)

            metric_history = workspace.mlflow_client.get_metric_history(mlflow_run.info.run_id, metric)
            if not metric_history:
                raise RuntimeError(f"No metric history found for {args.metric}.")

            latest_metric = metric_history[-1]

            workspace.mlflow_client.log_metric(
                mlflow_run.data.tags[MLFLOW_PARENT_RUN_ID],
                metric,
                latest_metric.value,
                step=trial.number,
            )

            return float(latest_metric.value)

        tango_settings = (
            TangoGlobalSettings.from_file(args.tango_settings) if args.tango_settings else TangoGlobalSettings.default()
        )

        for package_name in tango_settings.include_package or ():
            import_extra_module(package_name)

        initialize_logging(
            log_level=tango_settings.log_level,
            enable_cli_logs=True,
            file_friendly_logging=tango_settings.file_friendly_logging,
        )

        workspace = Workspace.from_params(tango_settings.workspace or {})
        if not isinstance(workspace, MLFlowWorkspace):
            raise ValueError("Tango workspace type must be mlflow.")

        optuna_settings = OptunaSettings.from_args(args)
        if optuna_settings.metric is None:
            raise ValueError("Optuna metric must be specified.")
        study_name = args.name or generate_unique_run_name(workspace.mlflow_client, workspace.experiment_name)

        with tempfile.TemporaryDirectory() as workdir:
            optuna_storage_path = os.path.join(workdir, "optuna.db")

            # Load previous study if exists
            mlflow_run = get_mlflow_run_by_optuna_study(
                workspace.mlflow_client,
                workspace.experiment_name,
                study_name=study_name,
            )
            if args.resume and mlflow_run is not None:
                logger.info(f"Resuming run {args.name}.")
                mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run.info.run_id,
                    artifact_path="optuna.db",
                    dst_path=workdir,
                )
                assert os.path.exists(optuna_storage_path)
            elif not args.resume and mlflow_run is not None:
                raise ValueError(f"Run {args.name} already exists. If you want to resume, use --resume option.")
            elif args.resume and mlflow_run is None:
                raise ValueError(f"Resume requested, but run {args.name} does not exist.")

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
                tango_settings.workspace = workspace.to_params().as_dict()

                mlflow.log_params(optuna_settings.to_params().as_flat_dict())

                objective = partial(
                    _objective,
                    metric=optuna_settings.metric,
                    experiment=args.experiment,
                    hparam_path=args.hparam_path,
                    tango_settings=tango_settings,
                )

                mlflow.log_artifact(args.experiment)
                mlflow.log_artifact(args.hparam_path)

                try:
                    study.optimize(
                        objective,
                        n_trials=args.n_trials,
                        timeout=args.timeout,
                    )
                    best_result = {
                        "trial": study.best_trial.number,
                        "params": study.best_trial.params,
                        "metric": study.best_trial.value,
                    }
                    best_result_filename = os.path.join(workdir, "best.json")
                    with open(best_result_filename, "w") as jsonfile:
                        json.dump(best_result, jsonfile)
                    mlflow.log_artifact(best_result_filename)
                finally:
                    mlflow.log_artifact(optuna_storage_path)
