#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for seq2seq, text to image.
Script adapted from run_summarization_flax.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import datasets
import jax
import jax.numpy as jnp
import optax
import transformers
import wandb
from datasets import Dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard_prng_key
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from transformers.models.bart.modeling_flax_bart import BartConfig

from dalle_mini.data import Dataset
from dalle_mini.model import CustomFlaxBartForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    image_vocab_size: Optional[int] = field(
        default=None,
        metadata={"help": "Vocab size of image encoder"},
    )
    image_length: Optional[int] = field(
        default=None,
        metadata={"help": "Number of tokens per image"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name_or_path"
        },
    )
    normalize_text: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to normalize text or not. By default, we refer to base model or don't normalize for new models."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    text_column: Optional[str] = field(
        default="caption",
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    encoding_column: Optional[str] = field(
        default="encoding",
        metadata={
            "help": "The name of the column in the datasets containing the image encodings."
        },
    )
    dataset_repo_or_path: str = field(
        default=None,
        metadata={"help": "The dataset repository containing encoded files."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (glob acceptable)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (glob acceptable)."},
    )
    dataset_type: str = field(
        default="datasets",
        metadata={"help": "Either 🤗 'dataset' (default) or 'webdataset'."},
    )
    # data loading should not be a bottleneck so we use "streaming" mode by default
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to stream the dataset."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the authentication token for private datasets."
        },
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing. Not used in streaming mode."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets. Not used in streaming mode."
        },
    )
    # default seed of None ensures we don't repeat the same items if script was interrupted during an epoch
    seed_dataset: int = field(
        default=None,
        metadata={
            "help": "Random seed for the dataset that will be set at the beginning of training."
        },
    )

    def __post_init__(self):
        if self.dataset_repo_or_path is None:
            raise ValueError("Need a dataset repository or path.")


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to training parameters.
    """

    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate."}
    )
    adafactor: bool = field(
        default=False,
        metadata={"help": "Whether or not to replace AdamW by Adafactor."},
    )
    weight_decay: float = field(
        default=None, metadata={"help": "Weight decay if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm for Adafactor."}
    )
    use_decay: bool = field(
        default=False,
        metadata={"help": "Whether to use decay in the learning rate scheduler."},
    )

    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )

    logging_steps: int = field(
        default=40, metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=400, metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=4000, metadata={"help": "Save checkpoint every X updates steps."}
    )
    log_model: bool = field(
        default=False,
        metadata={"help": "Log model to wandb at `save_steps` frequency."},
    )

    seed_model: int = field(
        default=42,
        metadata={
            "help": "Random seed for the model that will be set at the beginning of training."
        },
    )

    push_to_hub: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upload the trained model to the model hub after training."
        },
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Reference to a wandb artifact for resuming training."},
    )


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray = None
    epoch: int = 0
    train_time: float = 0.0  # total time the model trained
    train_samples: int = 0  # number of samples seen

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng)
        )

    def restore_state(self, artifact_dir):
        # restore optimizer state
        with (Path(artifact_dir) / "opt_state.msgpack").open("rb") as f:
            new_opt_state = from_bytes(self.opt_state, f.read())

        # restore other parameters
        with (Path(artifact_dir) / "training_state.json").open("r") as f:
            training_state = json.load(f)

        # replace state
        return self.replace(
            opt_state=new_opt_state,
            step=training_state["step"],
            train_time=training_state["train_time"],
            train_samples=training_state["train_samples"],
        )


def create_learning_rate_fn(
    num_warmup_steps: int,
    learning_rate: float,
    use_decay: bool,
    num_train_steps: int = None,  # used only with `use_decay`, typically train_size // batch_size * num_epochs
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    if use_decay:
        assert (
            num_train_steps is not None
        ), "Learning rate with decay requires number of training steps"
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    if not use_decay:
        return warmup_fn
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def wandb_log(metrics, step=None, prefix=None):
    if jax.process_index() == 0:
        log_metrics = {
            f"{prefix}/{k}" if prefix is not None else k: v for k, v in metrics.items()
        }
        if step is not None:
            log_metrics["train/step"] = step
        wandb.log(log_metrics)


def main():
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load dataset
    dataset = Dataset(
        **asdict(data_args),
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
    )

    # Set up wandb run
    wandb.init(
        entity="dalle-mini",
        project="dalle-mini",
        job_type="Seq2Seq",
        config=parser.parse_args(),
    )

    if training_args.resume_from_checkpoint is not None:
        artifact = wandb.run.use_artifact(training_args.resume_from_checkpoint)
        artifact_dir = artifact.download()

        # load model
        model = CustomFlaxBartForConditionalGeneration.from_pretrained(artifact_dir)
        # avoid OOM on TPU: see https://github.com/google/flax/issues/1658
        print(model.params)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            artifact_dir,
            use_fast=True,
        )

    else:
        # Set up our new model config
        # TODO: simplify with custom config class
        if model_args.config_name:
            config = BartConfig.from_pretrained(model_args.config_name)
        else:
            config = BartConfig.from_pretrained(model_args.model_name_or_path)
        if model_args.image_vocab_size:
            config.image_vocab_size = model_args.image_vocab_size
        assert (
            getattr(config, "image_vocab_size") is not None
        ), "image_vocab_size must be specified when not present in base model/config"
        if model_args.image_length:
            config.image_length = model_args.image_length
        assert (
            getattr(config, "image_length") is not None
        ), "image_length must be specified when not present in base model/config"
        # we append decoder bos to image vocab
        config.decoder_start_token_id = config.image_vocab_size
        # ensure we don't generate bos (in addition to decoder start token)
        config.force_bos_token_to_be_generated = False
        config.forced_bos_token_id = None  # we don't need this token
        config.forced_eos_token_id = None  # we don't need this token

        config.tie_word_embeddings = False
        config.min_length = config.image_length + 1
        config.max_length = config.image_length + 1

        # below tokens need to be set to avoid error during generation (converted to jnp.array)
        # they are not expected to be used and are set to unreachable token id
        config.bos_token_id = config.image_vocab_size + 1
        config.pos_token_id = config.image_vocab_size + 1
        config.eos_token_id = config.image_vocab_size + 1

        # save whether we normalize the text
        if model_args.normalize_text is not None:
            config.normalize_text = model_args.normalize_text
        else:
            config.normalize_text = getattr(config, "normalize_text", False)

        # Load or create new model
        if model_args.model_name_or_path:
            model = CustomFlaxBartForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                seed=training_args.seed_model,
                dtype=getattr(jnp, model_args.dtype),
            )
            # avoid OOM on TPU: see https://github.com/google/flax/issues/1658
            print(model.params)
        else:
            model = CustomFlaxBartForConditionalGeneration(
                config,
                seed=training_args.seed_model,
                dtype=getattr(jnp, model_args.dtype),
            )

        # Load tokenizer
        if model_args.tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, use_fast=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=True,
            )

    logger.info(f"TPUs: {jax.device_count()}")
    assert jax.device_count() == 8, "TPUs in use, please check running processes"

    # Preprocessing the datasets.
    # We need to normalize and tokenize inputs and targets.

    dataset.preprocess(
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        normalize_text=model.config.normalize_text,
    )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed_model)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * jax.device_count()
    )
    batch_size_per_update = train_batch_size * training_args.gradient_accumulation_steps
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    len_train_dataset, len_eval_dataset = dataset.length
    steps_per_epoch = (
        len_train_dataset // train_batch_size if len_train_dataset is not None else None
    )
    num_train_steps = (
        steps_per_epoch * num_epochs if steps_per_epoch is not None else None
    )

    # Create learning rate schedule
    learning_rate_fn = create_learning_rate_fn(
        training_args.warmup_steps,
        training_args.learning_rate,
        training_args.use_decay,
        num_train_steps,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBart.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_params = [
            (name, "scale")
            for name in [
                "self_attn_layer_norm",
                "layernorm_embedding",
                "final_layer_norm",
            ]
        ]
        flat_mask = {
            path: (path[-1] != "bias" and path[-2:] not in layer_norm_params)
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    if training_args.adafactor:
        # We use the default parameters here to initialize adafactor,
        # For more details about the parameters please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src/alias.py#L74
        optimizer = optax.adafactor(
            learning_rate=learning_rate_fn,
            weight_decay_rate=training_args.weight_decay,
            weight_decay_mask=decay_mask_fn,
            clipping_threshold=training_args.max_grad_norm,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # add gradient accumulation
    if training_args.gradient_accumulation_steps > 1:
        optimizer = optax.chain(
            optax.apply_every(training_args.gradient_accumulation_steps), optimizer
        )

    # Setup train state
    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )
    if training_args.resume_from_checkpoint is not None:
        # restore optimizer state and other parameters
        # we currently ignore partial epoch training: see https://github.com/borisdayma/dalle-mini/issues/105
        state = state.restore_state(artifact_dir)

    # label smoothed cross entropy
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    # Define gradient update step fn
    def train_step(state, batch, delta_time):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params, batch):
            labels = batch.pop("labels")
            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params, batch)
        grads = jax.lax.pmean(grads, "batch")
        state = state.apply_gradients(
            grads=grads,
            dropout_rng=new_dropout_rng,
            train_time=state.train_time + delta_time,
            train_samples=state.train_samples + train_batch_size,
        )

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return state, metrics

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & gradient accumulation) = {batch_size_per_update}"
    )
    epochs = tqdm(
        range(state.epoch, num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0
    )

    # set default x-axis as 'train/step'
    wandb_log({}, step=state.step)
    wandb.define_metric("*", step_metric="train/step")

    # add interesting config parameters
    wandb.config.update(
        {
            "len_train_dataset": len_train_dataset,
            "len_eval_dataset": len_eval_dataset,
            "batch_size_per_update": batch_size_per_update,
        }
    )

    # replicate state on each device
    state = state.replicate()

    def run_evaluation():
        # ======================== Evaluating ==============================
        eval_metrics = []
        if training_args.do_eval:
            eval_loader = dataset.dataloader("eval", eval_batch_size)
            eval_steps = (
                len_eval_dataset // eval_batch_size
                if len_eval_dataset is not None
                else None
            )
            for batch in tqdm(
                eval_loader,
                desc="Evaluating...",
                position=2,
                leave=False,
                total=eval_steps,
            ):
                # Model forward
                metrics = p_eval_step(state.params, batch)
                eval_metrics.append(metrics)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            # log metrics
            wandb_log(eval_metrics, step=unreplicate(state.step), prefix="eval")

            # Print metrics and update progress bar
            desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']})"
            epochs.write(desc)
            epochs.desc = desc

            return eval_metrics

    def run_save_model(state, eval_metrics=None):
        if jax.process_index() == 0:
            params = jax.device_get(unreplicate(state.params))
            # save model locally
            model.save_pretrained(
                training_args.output_dir,
                params=params,
            )

            # save tokenizer
            tokenizer.save_pretrained(training_args.output_dir)

            # save state
            opt_state = unreplicate(state.opt_state)
            with (Path(training_args.output_dir) / "opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(opt_state))
            state_dict = {
                k: jax.device_get(unreplicate(getattr(state, k))).item()
                for k in ["step", "epoch", "train_time", "train_samples"]
            }
            with (Path(training_args.output_dir) / "training_state.json").open(
                "w"
            ) as f:
                json.dump(
                    state_dict,
                    f,
                )

            # save to W&B
            if training_args.log_model:
                # save some space
                c = wandb.wandb_sdk.wandb_artifacts.get_artifacts_cache()
                c.cleanup(wandb.util.from_human_size("10GB"))

                metadata = dict(state_dict)
                if eval_metrics is not None:
                    metadata["eval"] = eval_metrics
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}", type="bart_model", metadata=metadata
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "flax_model.msgpack")
                )
                artifact.add_file(str(Path(training_args.output_dir) / "config.json"))
                artifact.add_file(
                    str(Path(training_args.output_dir) / "tokenizer.json")
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "tokenizer_config.json")
                )
                artifact.add_file(str(Path(training_args.output_dir) / "vocab.json"))
                artifact.add_file(str(Path(training_args.output_dir) / "merges.txt"))
                artifact.add_file(
                    str(Path(training_args.output_dir) / "special_tokens_map.json")
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "opt_state.msgpack")
                )
                artifact.add_file(
                    str(Path(training_args.output_dir) / "training_state.json")
                )

                wandb.run.log_artifact(artifact)

            # save to the hub
            if training_args.push_to_hub:
                model.save_pretrained(
                    training_args.output_dir,
                    params=params,
                    push_to_hub=training_args.push_to_hub,
                    commit_message=f"Saving weights and logs at step {unreplicate(state.step)+1}",
                    temp_dir=True,  # avoid issues with being in a repository
                )

    # init variables
    last_time = time.perf_counter()
    train_metrics = None

    for epoch in epochs:
        state.replace(epoch=jax_utils.replicate(epoch))
        # ======================== Training ================================
        wandb_log({"train/epoch": epoch}, step=unreplicate(state.step))

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = dataset.dataloader("train", train_batch_size)
        # train
        for batch in tqdm(
            train_loader,
            desc="Training...",
            position=1,
            leave=False,
            total=steps_per_epoch,
        ):

            # calculate delta time (we have a lag of one step but it's ok)
            new_time = time.perf_counter()
            delta_time = new_time - last_time
            last_time = new_time

            # train step
            state, train_metrics = p_train_step(
                state, batch, jax_utils.replicate(delta_time)
            )
            step = unreplicate(state.step)

            if step % training_args.logging_steps == 0 and jax.process_index() == 0:
                # log metrics
                metrics = unreplicate(train_metrics)
                # log state parameters
                state_dict = {
                    k.split("_")[-1]: unreplicate(getattr(state, k))
                    for k in ["epoch", "train_time", "train_samples"]
                }
                wandb_log({**metrics, **state_dict}, step=step, prefix="train")

            eval_metrics = None
            if training_args.eval_steps and step % training_args.eval_steps == 0:
                eval_metrics = run_evaluation()

            if step % training_args.save_steps == 0:
                run_save_model(state, eval_metrics)

        # log final train metrics
        if train_metrics is not None:
            train_metrics = unreplicate(train_metrics)
            wandb_log(train_metrics, step=step, prefix="train")

            epochs.write(
                f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metrics['loss']}, Learning Rate: {train_metrics['learning_rate']})"
            )

        # Final evaluation
        eval_metrics = run_evaluation()

        # save checkpoint after each epoch
        run_save_model(state, eval_metrics)


if __name__ == "__main__":
    main()
