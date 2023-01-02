import argparse
import dataclasses
import tempfile
from functools import partial
from typing import Optional

import mlflow
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
            "--metrics",
            type=str,
            help="The metrics you want to optimize.",
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
            metrics: Optional[str] = None
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
            def suggested_experiment() -> str:
                ...

            mlflow_run_name = f"{trial.study.study_name}_{trial.number}"
            with mlflow.start_run(nested=True, run_name=mlflow_run_name):
                mlflow.set_tag("optuna.study_name", trial.study.study_name)
                mlflow.set_tag("optuna.trial_number", trial.number)

                # TODO: implement optimaization

            return 0.0

        tango_settings = (
            TangoGlobalSettings.from_file(args.tango_settings) if args.tango_settings else TangoGlobalSettings.default()
        )
        workspace = Workspace.from_params(tango_settings.workspace or {})
        if not isinstance(workspace, MLFlowWorkspace):
            raise ValueError("Tango workspace type must be mlflow.")

        optuna_settings = (
            OptunaSettings.from_params(Params.from_file(args.optuna_settings), **vars(args))
            if args.optuna_settings
            else OptunaSettings(**vars(args))
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

        objective = partial(
            _objective,
            hparam_path=args.hparam_path,
            tango_settings=tango_settings,
        )

        with mlflow.start_run(
            run_id=mlflow_run.info.run_id if mlflow_run is not None else None,
            run_name=study_name if mlflow_run is None else None,
            tags={"job_type": RunKind.OPTUNA_STUDY.value},
        ):
            try:
                study.optimize(
                    objective,
                    n_trials=optuna_settings.n_trials,
                    timeout=optuna_settings.timeout,
                )
            finally:
                mlflow.log_artifact(optuna_storage_path, "optuna.db")
