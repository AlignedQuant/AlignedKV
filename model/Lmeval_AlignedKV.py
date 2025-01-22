import os
import transformers
from typing import Optional, Union
import torch
from lm_eval.models.huggingface import HFLM, eval_logger
from lm_eval import utils
from lm_eval.models.utils import stop_sequences_criteria, get_dtype
from huggingface_hub import HfApi
# from accelerate import Accelerator, DistributedType
from transformers.cache_utils import StaticCache, QuantizedCacheConfig, QuantizedCache, QuantoQuantizedCache, \
    HQQQuantizedCache
from model.KVCache_AlignedKV import QuantizedCache_AlignedKV
from model.Llama_AlignedKV import LlamaForCausalLM_AlignedKV


class LMEvalLlamaForCausalLM_AlignedKV(HFLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def _create_model(
            self,
            pretrained: str,
            revision: Optional[str] = "main",
            dtype: Optional[Union[str, torch.dtype]] = "auto",
            trust_remote_code: Optional[bool] = False,
            # arguments used for splitting a model across GPUs naively.
            # only used if `parallelize=True`.
            # (accelerate naive PP (device_map) options)
            parallelize: Optional[bool] = False,
            gpus: Optional[int] = None,
            max_memory_per_gpu: Optional[Union[int, str]] = None,
            max_cpu_memory: Optional[Union[int, str]] = None,
            offload_folder: Optional[str] = "./offload",
            # PEFT, delta weights and quantization options
            peft: Optional[str] = None,
            delta: Optional[str] = None,
            autogptq: Optional[Union[bool, str]] = False,
            **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        self.key_value_cache_class = model_kwargs["key_value_cache_class"]
        model_kwargs.pop("key_value_cache_class")
        if self.key_value_cache_class.lower() == "alignedkv":
            self.key_value_cache = QuantizedCache_AlignedKV
        elif self.key_value_cache_class.lower() == "static":
            self.key_value_cache = StaticCache
        elif self.key_value_cache_class.lower() == "kivi":
            self.key_value_cache = HQQQuantizedCache
            self.kvcache_config = QuantizedCacheConfig(axis_value=1, device=self.device)
        else:
            raise ValueError(f"Unsupported key-value cache class: {self.key_value_cache_class}")

        if not autogptq:
            self._model = LlamaForCausalLM_AlignedKV.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                # attn_implementation=attn_implementation,
                **model_kwargs,
            )
            # self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            #     pretrained,
            #     revision=revision,
            #     torch_dtype=get_dtype(dtype),
            #     trust_remote_code=trust_remote_code,
            #     **model_kwargs,
            # )
        else:
            raise NotImplementedError
        if peft:
            raise NotImplementedError
        elif delta:
            raise NotImplementedError

        return None

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        # # memroy occupy
        # print("current memory (byte):", torch.cuda.memory_allocated(device=None))
        # # print batchsize
        # print("bsz, seq: ", context.shape)
        # print("max_length")
        # kv-cache
        if self.key_value_cache_class.lower() == "kivi":
            KV_Cache = self.key_value_cache(self.kvcache_config)
        else:
            KV_Cache = self.key_value_cache(self.model.config, context.shape[0], max_length, self.device, torch.float16)
        result = self.model.generate(
            context,
            max_length=max_length,
            past_key_values=KV_Cache,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
        print("input:", context.shape)
        print(context)
        print("output:", result.shape)
        print(result)
        return result

    def get_model_info(self) -> dict:
        """
        Method to get Hugging Face model information for experiment reproducibility.
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return "float16"

        def get_model_sha(pretrained: str, revision: str) -> str:
            try:
                model_info = HfApi().model_info(repo_id=pretrained, revision=revision)
                return model_info.sha
            except Exception as e:
                eval_logger.warn(
                    f"Failed to get model SHA for {pretrained} at revision {revision}. Error: {e}"
                )
                return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        return model_info