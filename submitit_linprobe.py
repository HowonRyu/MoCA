# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import submitit


def parse_args():
    partial_parser = argparse.ArgumentParser("Partial parser", add_help=False)
    partial_parser.add_argument("--CV", default=False, action='store_true', help="Do 5-fold CV")
    #partial_parser.add_argument("--SCV_old", default=False, action='store_true', help="Do 5-fold CV (online)")
    args_partial, _ = partial_parser.parse_known_args()
    if args_partial.CV:
        import main_linprobe_CV as classification
    #elif args_partial.SCV_old:
    #    import main_linprobe_SCV_old as classification
    else:
        import main_linprobe as classification

    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE linear probe", parents=[classification_parser])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=43200000, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--wk_dir", default="", type=str, help="Working directory for local machine") # changed - added
    parser.add_argument("--job_name", default=None, type=str, help="Job nickname for checkpoint name") # changed - added

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    parser.add_argument("--wait", action='store_true', help="Waits for submitit results (for k8s jobs)")
    parser.add_argument("--CV", action='store_true', help="Do 5-fold CV")
    #parser.add_argument("--SCV_old", default=False, action='store_true', help="Do subject-level 5-fold CV (online)")
    return parser.parse_args(), classification

def get_shared_folder(wk_dir_path) -> Path: # changed - for running on local machine (CPU)
    home_directory = Path(wk_dir_path)  # changed - path
    if home_directory.is_dir():
        p = home_directory / "experiments"
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")



def get_init_file(wk_dir_path):  # changed - for running on local machine (CPU)
    os.makedirs(str(get_shared_folder(wk_dir_path)), exist_ok=True)   # changed -job_dir_path
    init_file = get_shared_folder(wk_dir_path) / f"{uuid.uuid4().hex}_init" # changed -job_dir_path
    if init_file.exists():
        os.remove(str(init_file))
    return init_file



class Trainer(object):
    def __init__(self, args, classification):
        self.args = args
        self.classification = classification

    def __call__(self):
        self._setup_gpu_args()
        self.classification.main(self.args)    

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(wk_dir_path=self.args.wk_dir).as_uri()  #changed - wk_dir_path=self.args.wk_dir,
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", f"{job_env.job_id}({self.args.job_name})")) # changed - changed from  f"{job_env.job_id}"
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args, classification = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder(wk_dir_path=args.wk_dir) / "%j" # changed - wk_dir_path=args.wk_dir

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="mae")

    args.dist_url = get_init_file(wk_dir_path=args.wk_dir).as_uri() #changed - wk_dir_path=args.wk_dir
    args.output_dir = args.job_dir

    trainer = Trainer(args, classification)
    job = executor.submit(trainer)
    print("Submitted job_id:", job.job_id)


    if args.wait:
        job.result()

if __name__ == "__main__":
    main()
