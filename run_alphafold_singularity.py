#!/usr/bin/env python3

# Script to run Alphafold 2.3.0 using Singularity.
# Builds the command and executes it, using a Alphafold image hosted on Dockerhub.
#
# Author: Diego Alvarez S. [dialvarezs@gmail.com]
# Last modified: 2022-12-13

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple

CONTAINER_IMAGE = "docker://catgumag/alphafold:2.3.0"
ROOT_MOUNT_DIRECTORY = "/mnt"


def main():
    args = parse_arguments()

    data_path = Path(args.data_dir)

    # Path to the Uniref90 database for use by JackHMMER.
    uniref90_database_path = data_path / "uniref90" / "uniref90.fasta"

    # Path to the Uniprot database for use by JackHMMER.
    uniprot_database_path = data_path / "uniprot" / "uniprot.fasta"

    # Path to the MGnify database for use by JackHMMER.
    mgnify_database_path = data_path / "mgnify" / "mgy_clusters_2022_05.fa"

    # Path to the BFD database for use by HHblits.
    bfd_database_path = (
        data_path / "bfd" / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
    )

    # Path to the Small BFD database for use by JackHMMER.
    small_bfd_database_path = (
        data_path / "small_bfd" / "bfd-first_non_consensus_sequences.fasta"
    )

    # Path to the Uniref30 database for use by HHblits.
    uniref30_database_path = data_path / "uniref30" / "UniRef30_2021_03"

    # Path to the PDB70 database for use by HHsearch.
    pdb70_database_path = data_path / "pdb70" / "pdb70"

    # Path to the PDB seqres database for use by hmmsearch.
    pdb_seqres_database_path = data_path / "pdb_seqres" / "pdb_seqres.txt"

    # Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
    template_mmcif_dir = data_path / "pdb_mmcif" / "mmcif_files"

    # Path to a file mapping obsolete PDB IDs to their replacements.
    obsolete_pdbs_path = data_path / "pdb_mmcif" / "obsolete.dat"

    mounts = []
    # Mount each fasta path as a unique target directory
    target_fasta_paths = []
    for i, fasta_path in enumerate(args.fasta_paths):
        mount, target_path = _generate_mount(f"fasta_path_{i}", Path(fasta_path))
        mounts.append(mount)
        target_fasta_paths.append(target_path)
    command_args = [f"--fasta_paths={','.join(target_fasta_paths)}"]
    # Mount database and output directories
    database_paths = [
        ("uniref90_database_path", uniref90_database_path),
        ("mgnify_database_path", mgnify_database_path),
        ("data_dir", args.data_dir),
        ("template_mmcif_dir", template_mmcif_dir),
        ("obsolete_pdbs_path", obsolete_pdbs_path),
    ]
    if args.model_preset == "multimer":
        database_paths.append(("uniprot_database_path", uniprot_database_path))
        database_paths.append(("pdb_seqres_database_path", pdb_seqres_database_path))
    else:
        database_paths.append(("pdb70_database_path", pdb70_database_path))

    if args.db_preset == "reduced_dbs":
        database_paths.append(("small_bfd_database_path", small_bfd_database_path))
    else:
        database_paths.extend(
            [
                ("uniref30_database_path", uniref30_database_path),
                ("bfd_database_path", bfd_database_path),
            ]
        )

    for name, path in database_paths:
        if path:
            mount, target_path = _generate_mount(name, Path(path))
            mounts.append(mount)
            command_args.append(f"--{name}={target_path}")

    output_mount, output_target_path = _generate_mount(
        "output", Path(args.output_dir), read_only=False
    )
    mounts.append(output_mount)

    # Set general options for the alphafold script
    command_args.extend(
        [
            f"--output_dir={output_target_path}",
            f"--max_template_date={args.max_template_date}",
            f"--db_preset={args.db_preset}",
            f"--model_preset={args.model_preset}",
            f"--benchmark={args.benchmark}",
            f"--use_precomputed_msas={args.use_precomputed_msas}",
            f"--num_multimer_predictions_per_model={args.num_multimer_predictions_per_model}",
            f"--run_relax={args.run_relax}",
            f"--use_gpu_relax={args.enable_gpu_relax}",
            "--logtostderr",
        ]
    )

    # Set environment variables for the container
    env = {
        "NVIDIA_VISIBLE_DEVICES": args.gpu_devices,
        # The following flags allow us to make predictions on proteins that
        # would typically be too long to fit into GPU memory.
        "TF_FORCE_UNIFIED_MEMORY": "1",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0",
        "OPENMM_CPU_THREADS": args.cpus,
        "MAX_CPUS": args.cpus,
    }

    # Generate the final command to execute
    command = [
        "singularity",
        "exec",
        "--nv" if args.use_gpu else "",
        "--bind",
        ",".join(mounts),
        *[f'--env="{k}={v}"' for k, v in env.items()],
        args.docker_image,
        "/app/run_alphafold.sh",
        *command_args,
    ]

    print("Executing: " + " ".join(command))

    p = subprocess.run(command)
    p.check_returncode()


def _generate_mount(mount_name: str, path: Path, read_only=True) -> Tuple[str, str]:
    """
    Generate a mount line for a singularity container.
    :param mount_name: The name of the mount point.
    :param path: The path to mount.
    :return: A tuple of the mount line and the path to mount.
    """
    path = path.resolve()
    source_path = path.parent
    target_path = Path(ROOT_MOUNT_DIRECTORY) / mount_name
    opts = "ro" if read_only else "rw"

    mount_cmd = f"{source_path}:{target_path}:{opts}"
    return mount_cmd, str(target_path / path.name)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Singularity launch script for Alphafold v2.2.2"
    )

    parser.add_argument(
        "--fasta-paths",
        "-f",
        required=True,
        nargs="+",
        help="Paths to FASTA files, each containing one sequence. "
        "All FASTA paths must have a unique basename as the basename "
        "is used to name the output directories for each prediction.",
    )
    parser.add_argument(
        "--max-template-date",
        "-t",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Maximum template release date to consider "
        "(ISO-8601 format - i.e. YYYY-MM-DD). "
        "Important if folding historical test sets.",
    )
    parser.add_argument(
        "--db-preset",
        choices=["reduced_dbs", "full_dbs"],
        default="full_dbs",
        help="Choose preset model configuration - no ensembling with "
        "uniref90 + bfd + uniclust30 (full_dbs), or "
        "8 model ensemblings with uniref90 + bfd + uniclust30 (casp14).",
    )
    parser.add_argument(
        "--model-preset",
        choices=["monomer", "monomer_casp14", "monomer_ptm", "multimer"],
        default="monomer",
        help="Choose preset model configuration - the monomer model, the monomer model "
        "with extra ensembling, monomer model with pTM head, or multimer model",
    )
    parser.add_argument(
        "--num-multimer-predictions-per-model",
        default=5,
        type=int,
        help="How many "
        "predictions (each with a different random seed) will be "
        "generated per model. E.g. if this is 2 and there are 5 "
        "models then there will be 10 predictions per input. "
        "Note: this FLAG only applies if model_preset=multimer",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        default=False,
        action="store_true",
        help="Run multiple JAX model evaluations to obtain a timing "
        "that excludes the compilation time, which should be more indicative "
        "of the time required for inferencing many proteins.",
    )
    parser.add_argument(
        "--use-precomputed-msas",
        default=False,
        action="store_true",
        help="Whether to read MSAs that have been written to disk instead of running "
        "the MSA tools. The MSA files are looked up in the output directory, so it "
        "must stay the same between multiple runs that are to reuse the MSAs. "
        "WARNING: This will not check if the sequence, database or configuration "
        "have changed.",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="./databases/",
        help="Path to directory with supporting data: AlphaFold parameters and genetic "
        "and template databases. Set to the target of download_all_databases.sh.",
    )
    parser.add_argument(
        "--docker-image", default=CONTAINER_IMAGE, help="Alphafold docker image."
    )
    parser.add_argument(
        "--output-dir", "-o", default="results/", help="Output directory for results."
    )
    parser.add_argument(
        "--use-gpu",
        default=True,
        action="store_true",
        help="Enable NVIDIA runtime to run with GPUs.",
    )
    parser.add_argument(
        "--run-relax",
        default=True,
        action="store_true",
        help="Whether to run the final relaxation step on the predicted models. Turning "
        "relax off might result in predictions with distracting stereochemical "
        "violations but might help in case you are having issues with the "
        "relaxation stage.",
    )
    parser.add_argument(
        "--enable-gpu-relax",
        default=True,
        action="store_true",
        help="Run relax on GPU if GPU is enabled.",
    )
    parser.add_argument(
        "--gpu-devices",
        default="all",
        help="Comma separated list of devices to pass to NVIDIA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--cpus", "-c", type=int, default=8, help="Number of CPUs to use."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
