from abc import ABC, abstractmethod
from argparse import Namespace
from typing import NamedTuple, List
import pandas as pd
from botocore.exceptions import ClientError
import boto3
import os

from config import PATH


class ResumeData(NamedTuple):
    # This bool indicates whether or not we are resuming an experiment or
    # starting from scratch
    resume: bool
    current_steps: int
    current_episodes: int
    model_paths: List[str]


def build_resume_message(completed_steps: int, total_steps: int, completed_episodes: int, total_episodes: int,
                         log_file: str, model_paths: List[str]) -> str:
    msg = 'Resuming experiment:\n'
    msg += '-'*len('Resuming experiment:') + '\n'

    if total_steps < float('inf'):
        msg += '{} out {} steps completed.\n'.format(completed_steps, total_steps)

    if total_episodes < float('inf'):
        msg += '{} out {} steps completed.\n'.format(completed_episodes, total_episodes)

    msg += 'Pre-existing log file found at {}\n'.format(log_file)

    msg += 'Pre-existing model files found at [\n'

    for m in model_paths:
        msg += "    '{}',\n".format(m)

    msg += ']'

    return msg


class Resume(ABC):
    """Abstract for classes that handle resuming experiments from a checkpoint."""
    @staticmethod
    @abstractmethod
    def _resume_models(args: Namespace, repeat: int) -> (bool, List[str], float):
        """Gets models to resume experiment with.

        Args:
            args: Command line arguments
            repeat: Repeat index of run within experiment

        Returns:
            models_found: A bool indicating whether or not any models were found
            model_files: A list of filepaths to the models
            num_completed_steps: The number of steps that each model has been trained with
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _resume_log(args: Namespace, repeat: int) -> (bool, List[str], float, float):
        """Gets logs to resume experiment with.

        Args:
            args: Command line arguments
            repeat: Repeat index of run within experiment

        Returns:
            logs_found: A bool indicating whether or not
            log_file: Filepath to the logs for this run
            num_completed_steps: The number of steps that each model has been trained with
            num_completed_episodes: The number of episodes that have been completed
        """
        raise NotImplementedError

    def resume(self, args: Namespace, repeat: int) -> ResumeData:
        logs_found, log_file, num_completed_steps, num_completed_episodes = self._resume_log(args, repeat)
        # Use `num_completed_steps` from the model file
        models_found, model_files, num_completed_steps = self._resume_models(args, repeat)

        # This operator is XOR. I know, I hadn't used this in python myself
        if logs_found ^ models_found:
            raise RuntimeError('Repeat {}: logs_found={}, models_found={}.'.format(repeat, logs_found, models_found))
        elif logs_found and models_found:
            # Trim the log file to the number of model steps
            df = pd.read_csv(log_file, comment='#')
            df[df['steps'] <= num_completed_steps].to_csv(log_file, index=False)

            msg = build_resume_message(num_completed_steps, args.total_steps, num_completed_episodes,
                                       args.total_episodes, log_file, model_files)
            print(msg)
        else:
            print('Neither logs nor models found: starting experiment from scratch.')

        return ResumeData(
            logs_found and models_found,
            num_completed_steps,
            num_completed_episodes,
            model_files
        )


def get_latest_complete_set_of_models(n_models: int, repeat: int, checkpoint_models: List[dict]) -> (List[str], float):
    """This handles the edge case where a training exited after saving only 1 of N models.

    In this case we want to restart from the last checkpoint which has all N models.
    """
    max_steps = max(checkpoint_models, key=lambda x: x['steps'])['steps']

    latest_models = [f['filepath'] for f in checkpoint_models if f['steps'] == max_steps]

    if len(latest_models) == n_models:
        return latest_models, max_steps
    elif len(latest_models) < n_models:
        # Trim the incomplete models and try again
        # #RecursionFlex
        checkpoint_models = [i for i in checkpoint_models if f['steps'] < max_steps]
        return get_latest_complete_set_of_models(n_models, repeat, checkpoint_models)
    else:
        # More models than expected, somethings really wrong
        raise RuntimeError('Model resuming found more models than expected!'
                           'Expected = {}, Found = {}'.format(n_models, len(latest_models)))


class LocalResume(Resume):
    @staticmethod
    def _resume_models(args: Namespace, repeat: int) -> (bool, List[str], float):
        if os.path.exists(f'{PATH}/experiments/{args.save_folder}/models/'):
            # Get latest checkpoint
            checkpoint_models = []
            for root, _, files in os.walk(f'{PATH}/experiments/{args.save_folder}/models/'):
                for f in sorted(files):
                    model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
                    if int(model_args['repeat']) == repeat:
                        checkpoint_models.append({
                            'species': int(model_args['species']),
                            'steps': float(model_args['steps']),
                            'repeat': int(model_args['repeat']),
                            'filepath': f
                        })

            if not checkpoint_models:
                # No models found
                return False, [], 0

            latest_models, num_completed_steps = get_latest_complete_set_of_models(args.n_species, repeat,
                                                                                   checkpoint_models)
            latest_models = [os.path.join(root, f) for f in latest_models]
            args.agent_location = latest_models
            return bool(latest_models), latest_models, num_completed_steps
        else:
            return False, [], 0

    @staticmethod
    def _resume_log(args: Namespace, repeat: int):
        save_file = f'repeat={repeat}'
        experiment_folder_exists = os.path.exists(f'{PATH}/experiments/{args.save_folder}')
        repeat_exists = os.path.exists(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv')
        if experiment_folder_exists and repeat_exists:
            old_log_file = pd.read_csv(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', comment='#')
            logs_found = True
            num_completed_steps = old_log_file.iloc[-1].steps
            num_completed_episodes = old_log_file.iloc[-1].episodes
        else:
            logs_found = False
            num_completed_steps = 0
            num_completed_episodes = 0

        return logs_found, f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', num_completed_steps, num_completed_episodes


class S3Resume(Resume):
    @staticmethod
    def _resume_log(args: Namespace, repeat: int):
        save_file = f'repeat={repeat}'
        # The easiest way to check if an experiment checkpoint exists is just to try to
        # load it and check for a 404.
        try:
            s3 = boto3.client('s3')
            os.makedirs(f'{PATH}/experiments/{args.save_folder}/logs/', exist_ok=True)
            s3.download_file(
                args.s3_bucket,
                f'{args.save_folder}/logs/{save_file}.csv',
                f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv'
            )
            old_log_file = pd.read_csv(f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', comment='#')

            logs_found = True
            num_completed_steps = old_log_file.iloc[-1].steps
            num_completed_episodes = old_log_file.iloc[-1].episodes
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                # Object not found
                logs_found = False
                num_completed_steps = 0
                num_completed_episodes = 0
            else:
                raise e
        except (FileNotFoundError, PermissionError) as e:
            # Another path for not finding the file
            # For some reason this raises a permission error as well
            logs_found = False
            num_completed_steps = 0
            num_completed_episodes = 0

        return logs_found, f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv', num_completed_steps, num_completed_episodes

    @staticmethod
    def _resume_models(args: Namespace, repeat: int) -> (bool, List[str], float):
        s3 = boto3.client('s3')
        checkpoint_models = []
        object_query = s3.list_objects(Bucket=args.s3_bucket, Prefix=f'{args.save_folder}/models/')

        if 'Contents' in object_query:
            for key in object_query['Contents']:
                f = key['Key'].split('/')[-1]
                model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
                if int(model_args['repeat']) == repeat:
                    checkpoint_models.append({
                        'species': int(model_args['species']),
                        'steps': float(model_args['steps']),
                        'repeat': int(model_args['repeat']),
                        'filepath': f
                    })

            if not checkpoint_models:
                # No models found
                return False, [], 0

            latest_models, num_completed_steps = get_latest_complete_set_of_models(args.n_species, repeat, checkpoint_models)
            latest_models = [f's3://{args.s3_bucket}/{args.save_folder}/models/{f}' for f in latest_models]
            args.agent_location = latest_models
            return bool(latest_models), latest_models, num_completed_steps
        else:
            return False, [], 0


"""
These two resume methods can be tested with the following commands:

# Local resume multi-repeats
python main.py --env laser --n-envs 128 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 3000 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test-multi-repeats --n-repeats 4 --n-processes 1 \
    --model-interval 1000

python main.py --env laser --n-envs 128 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 5000 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test-multi-repeats --n-repeats 4 --n-processes 1 \
    --model-interval 1000


# S3 resume multi-repeats
python main.py --env laser --n-envs 128 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 3000 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test-multi-repeats --n-repeats 4 --n-processes 1 \
    --s3-bucket oscarknagg-experiments --s3-interval 1000 --model-interval 1000 --resume-mode s3

python main.py --env laser --n-envs 128 --n-agents 2 --n-species 2 --height 9 --width 9 \
    --total-steps 5000 --agent-type gru --entropy 0.01 --lr 5e-4 --gamma 0.99 --mask-dones True \
    --obs-rotate True --obs-in-front 11 --obs-side 6 --obs-behind 2 \
    --laser-tag-map small2 --save-folder test-multi-repeats --n-repeats 4 --n-processes 1 \
    --s3-bucket oscarknagg-experiments --s3-interval 10000 --model-interval 1000 --resume-mode s3



"""
