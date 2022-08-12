import os
import tensorflow as tf
from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str

###
loaded = tf.saved_model.load("/workspace/playground/ssd/after_nms")
infer = loaded.signatures["serving_default"]
# print(loaded.signatures)
# print(infer)

###

backend = "cpu"
backend_config = "local-task"
args = ["--iree-llvm-target-cpu-features=host"]

model_path = "/workspace/playground/ssd/after_nms"
compiler_module = tfc.compile_saved_model(model_path, import_type="SIGNATURE_DEF", saved_model_tags=set(["serve"]), import_only=False, target_backends=[backend])
flatbuffer_blob = compile_str(compiler_module, input_type="mhlo", target_backends=[backend], extra_args=args)

config = ireert.Config(backend_config)
ctx = ireert.SystemContext(config=config)

# Save module as MLIR file in a directory
vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)

#tracer = ireert.Tracer(os.getcwd())
# TODO: Remove printing of "Tracing module.predict"
ctx.add_vm_module(vm_module)
BertCompiled = ctx.modules.module

#Dump module
ARITFACTS_DIR = os.getcwd()
mlir_path = os.path.join(ARITFACTS_DIR, "model.mlir")
with open(mlir_path, "wt") as output_file:
    output_file.write(compiler_module.decode('utf-8'))
print(f"Wrote MLIR to path '{mlir_path}'")

